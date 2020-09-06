#include "neural/config.h"
#include "neural/node.h"

#pragma once

namespace mz {

template<class State, class Environment, class Game>
struct BaseGame {

	BaseGame() { history.push_back(environment.get_state()); };

	size_t size() const { return actions.size(); }

	void apply(Action action)
	{
		auto reward = environment.act(action);

		history.push_back(environment.get_state());
		actions.push_back(action);
		rewards.push_back(reward);
	}
	void store_search_statistics(const NodePtr &root)
	{
		auto &arr = child_visits.emplace_back();

		for (auto &[action, child] : root->children)
			arr[Game::to_nn_idx(action)] = static_cast<float>(child->n) / root->n;

		root_values.push_back(root->get_q());
	}
	std::vector<Target> make_target(const size_t state_idx)
	{
		std::vector<Target> targets;
		auto last_idx = std::min(state_idx + NUM_UNROLL_STEPS, size());

		for (size_t i = state_idx; i < last_idx; ++i) {

			float value = 0.0;
			auto bootstrap_index = i + TD_STEPS;

			if (bootstrap_index < size())
				value = root_values[bootstrap_index] * std::pow(DISCOUNT, TD_STEPS);

			for (int j = i+1, exp = 0; j < std::min(bootstrap_index, size()); ++j, ++exp)
				value += rewards[j] * std::pow(DISCOUNT, exp);

			targets.emplace_back(
				torch::tensor(value, DEVICE).unsqueeze_(0), 
				torch::tensor(rewards[i], DEVICE).unsqueeze_(0), 
				torch::tensor(at::ArrayRef<float>(child_visits[i]), DEVICE).unsqueeze_(0));
		}
		return targets;
	}

	int to_play() const { return environment.to_play(); }
	bool terminal() const { return environment.terminal(); }
	Action ask_input() const { return environment.ask_input(); }
	ActionList legal_actions() const { return environment.legal_actions(); }
	ActionList get_actions(size_t first) const
	{
		auto last = std::min(first + NUM_UNROLL_STEPS, size());
		return ActionList(actions.begin() + first, actions.begin() + last);
	}
	
protected:
	Environment environment;
	std::vector<State> history;
	std::vector<Action> actions;
	std::vector<float> rewards = {0.0};
	std::vector<float> root_values;
	std::vector<std::array<float, ACTION_SPACE_SIZE>> child_visits;
};

}