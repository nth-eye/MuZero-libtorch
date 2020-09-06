#include <random>
#include <limits>
#include <ranges>

#include "config.h"
#include "node.h"

#pragma once

namespace mz {

// Support vector for scalar-support conversion.
static const auto support = torch::linspace(-SUPPORT_SIZE, SUPPORT_SIZE + 1, 1.0);
// Parameter for scalar-support.
static const float e = 0.0001;
// Function to get current temperature.
static float temperature_fn(int num_moves)
{
	return num_moves > 30 ? 1.0 : 0.0;
}

struct NetworkOutput {
	torch::Tensor value;
	torch::Tensor reward;
	torch::Tensor policy_logits;
	torch::Tensor hidden_state;
};

struct NetworkOptions {
	int in_shape_f[3] = {256, 6, 6};
	int in_shape_g[3] = {256, 6, 6};
	int in_shape_h[3] = {128, 96, 96};

	int filters_f = 256;
	int filters_g = 256;
	int filters_h = 256;

	int num_blocks_f = 20;
	int num_blocks_g = 16;
	int num_blocks_h = 16;

	int reduced_p = 2;
	int reduced_v = 1;
	int reduced_r = 1;

	bool downsample = true;
};

struct MinMaxStats {

	MinMaxStats(Bounds bounds) { if (bounds) std::tie(min, max) = bounds.value(); }

	void update(float value)
	{
		max = std::max(max, value);
		min = std::min(min, value);
	}
	float normalize(float value) const
	{
		if (max > min) return (value - min) / (max - min);
		return value;
	}

private:
	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::lowest();
};

inline torch::Tensor scale_gradient(torch::Tensor x, const float scale)
{
	return x * scale + x.detach() * (1 - scale);
}

inline torch::Tensor softmax_cross_entropy_with_logits(torch::Tensor logits, torch::Tensor labels)
{
	return torch::mean(-torch::sum(labels * torch::log_softmax(logits, 1), 1));
}

inline torch::Tensor scalar_loss(torch::Tensor x, torch::Tensor label) 
{
	if constexpr (CATEGORICAL_LOSS)
		return softmax_cross_entropy_with_logits(x, label);
	else
		return torch::mse_loss(x, label); 
}

// FIXME
inline torch::Tensor support_to_scalar(torch::Tensor x)
{
	auto probs = torch::softmax(x, 0);
	auto supp = support.expand(probs.sizes());

	x = torch::sum(supp * probs, 0);

	x = torch::sign(x) * (
			torch::square(
				(torch::sqrt(1 + 4*e*(torch::abs(x) + 1 + e)) - 1) 
				/ (2 * e)
			) - 1
		);

	return x;
}

// FIXME
inline torch::Tensor scalar_to_support(torch::Tensor x)
{
	x = torch::sign(x) * (torch::sqrt(torch::abs(x) + 1) - 1) + e*x;

	return x;
}

inline void add_exploration_noise(const NodePtr &node)
{
	static thread_local std::mt19937 rng;
	std::gamma_distribution<float> dist(ALPHA, 1.0f);
	std::vector<float> dirichlet_noise(node->children.size(), 0.0f);

	float sum = 0.0f;

	for (auto &it : dirichlet_noise)
		sum += (it = dist(rng));

	for (auto &it : dirichlet_noise)
		it /= sum + 1e-8f;

	for (int i = 0; i < node->children.size(); ++i) {
		node->children[i].second->p *= (1.0f - EPSILON);
		node->children[i].second->p += dirichlet_noise[i] * EPSILON;
	}
}

inline float ucb_score(const Node *parent, const NodePtr &child, const MinMaxStats &min_max_stats)
{
	float c_puct = PB_C_INIT + std::log((parent->n + PB_C_BASE + 1) / PB_C_BASE);
	
	float prior_score = child->get_p() * c_puct * std::sqrt(parent->n);
	float value_score = 0.0;

	if (child->n > 0)
		value_score = child->r + DISCOUNT * min_max_stats.normalize(child->get_q());

	return prior_score + value_score;
}

inline Action select_action(const NodePtr &node, const int num_moves)
{
	static thread_local std::mt19937 rng;
	std::vector<int> counts;

	for (auto &[action, child] : node->children)
		counts.push_back(child->n);

	const float exponent = 1.0 / temperature_fn(num_moves);

	// If temperature is close to zero.
	if (exponent > 1e6)
		return node->children[std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()))].first;

	std::vector<float> probs; probs.reserve(counts.size());
	float sum_probs = 0.0;

	for (auto n : counts) {
		auto res = std::pow(n, exponent);
		sum_probs += res;
		probs.push_back(res);
	}

	for (auto &p : probs)
		p /= sum_probs;

	return node->children[std::discrete_distribution<int>(probs.begin(), probs.end())(rng)].first;
}

inline std::pair<Action, Node*> select(const Node *node, MinMaxStats &min_max_stats)
{
	float best_value = -std::numeric_limits<float>::infinity();
	Action best_action = -1;
	Node *best_child = nullptr;

	for (auto &[act, child] : node->children) {

		float uct = ucb_score(node, child, min_max_stats);
		
		if (uct > best_value) {
			best_value = uct;
			best_action = act;
			best_child = child.get();
		}
	}
	return {best_action, best_child};
}

inline void expand(Node *node, int turn, NetworkOutput &output, const ActionList &actions)
{
	node->r = output.reward.item<float>();
	node->turn = turn;
	node->state = std::move(output.hidden_state);

	output.policy_logits = output.policy_logits.softmax(1);

	for (int i = 0; i < actions.size(); ++i)
		node->children.emplace_back(actions[i], std::make_unique<Node>(output.policy_logits[0][i].item<float>()));
}

inline void backpropagate(const std::vector<Node*> &search_path, float value, const int turn, MinMaxStats &min_max_stats)
{
	for (auto node : std::ranges::reverse_view(search_path)) {
		node->n += 1;
		node->w += node->turn == turn ? value : -value;
		min_max_stats.update(node->get_q());
		value = node->r + DISCOUNT * value;
	}
}

}