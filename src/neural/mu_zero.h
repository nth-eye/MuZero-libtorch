#include <atomic>
#include <chrono>

#include "network.h"
#include "replay_buffer.h"

#pragma once

namespace mz {

template<class Game>
struct MuZero {

	MuZero(const NetworkOptions &net_op_, bool restore, const std::string &in_weights, const std::string &out_weights);

	void start_train();

	void test_human();
	void test_self();
	void test_elo();

	float update_weights(const std::vector<Sample> &batch);

	void train_network();
	void selfplay();
	Game play_game(MuZeroNetwork network);

	template<class ActionHistory>
	void run_mcts(MuZeroNetwork network, const NodePtr &root, ActionHistory history);

private:
	// Delete all illegal actions from policy tensor.
	torch::Tensor filter_policy(torch::Tensor &policy, const ActionList &actions)
	{
		auto new_policy = torch::empty({1, int(actions.size())});
		auto pol_ac = new_policy.template accessor<float, 2>();

		for (int i = 0; i < actions.size(); ++i)
			pol_ac[0][i] = policy[0][Game::to_nn_idx(actions[i])].template item<float>();

		return new_policy;
	}

	MinMaxStats min_max_stats;
	MuZeroNetwork main_network;
	ReplayBuffer<Game> replay_buffer;

	NetworkOptions net_op;

	std::string in_weights_file;
	std::string out_weights_file;

	torch::optim::SGD optimizer;

	std::atomic_bool train_flag = false;
	std::atomic_uint games_played = 0;
};


template<class Game>
MuZero<Game>::MuZero(const NetworkOptions &net_op_, bool restore, const std::string &in_weights, const std::string &out_weights) :
	min_max_stats(KNOWN_BOUNDS),
	main_network(net_op_),
	net_op(net_op_),
	in_weights_file(in_weights),
	out_weights_file(out_weights),
	optimizer(main_network->parameters(), torch::optim::SGDOptions(LR_INIT).weight_decay(WEIGHT_DECAY).momentum(MOMENTUM))
{
	main_network->to(DEVICE);

	if (restore)
		torch::load(main_network, in_weights_file);
	torch::save(main_network, out_weights_file);
}

template<class Game>
void MuZero<Game>::start_train()
{
	train_flag = true;

	std::vector<std::jthread> threads; threads.reserve(NUM_ACTORS);

	for (int i = 0; i < NUM_ACTORS; ++i)
		threads.emplace_back(&MuZero::selfplay, this);

	train_network();
}

template<class Game>
void MuZero<Game>::test_human()
{
	auto game = Game();

	int first = game.to_play();
	int player = first;

	while (!game.terminal()) {

		game.display();
		Action action = Game::NO_MOVE;

		if (player == first) {
			std::cout << "\nYour turn: " << std::flush;
			action = game.ask_input();
		} else {
			auto root = std::make_unique<Node>(1.0f);
			auto img = game.make_image();
			auto output = main_network->initial_inference(img);
			auto legal_actions = game.legal_actions();

			output.policy_logits = filter_policy(output.policy_logits, legal_actions);
			expand(root.get(), game.to_play(), output, legal_actions);	
			run_mcts(main_network, root, game.action_history());

			action = select_action(root, game.size());
			std::cout << "\nMuZero has moved" << std::endl;
		}
		game.apply(action);
		player = game.to_play();
	}
	game.display();
}

template<class Game>
float MuZero<Game>::update_weights(const std::vector<Sample> &batch)
{
	auto loss = torch::zeros(1, DEVICE);
	main_network->zero_grad();

	for (auto &[image, actions, targets] : batch) {

		auto [value, reward, policy_logits, hidden_state] = main_network->initial_inference(image);
		std::vector<std::tuple<float, torch::Tensor, torch::Tensor, torch::Tensor>> predictions;
		predictions.reserve(actions.size());// = {{1.0, value, reward, policy_logits}};

		std::cout << "image:\n" << image << std::endl;
		std::cout << "reward:\n" << reward << std::endl;
		std::cout << "policy_logits:\n" << policy_logits << std::endl;
		//std::cout << "hidden_state:\n" << hidden_state << std::endl;

		for (const auto action : actions) {
			auto input_to_dynamics = Game::make_next_state(hidden_state, action);
			auto o = main_network->recurrent_inference(input_to_dynamics);
			predictions.emplace_back(1.0 / actions.size(), o.value, o.reward, o.policy_logits);

			hidden_state = scale_gradient(o.hidden_state, 0.5);
		}
		for (size_t i = 0; i < predictions.size(); ++i) {

			auto [gradient_scale, value, reward, policy_logits] = predictions[i];
			auto [target_value, target_reward, target_policy] = targets[i];

			std::cout << "Predicted policy:\n" << policy_logits << "\nTarget policy:\n" << target_policy << std::endl;

			auto l1 = scalar_loss(value, target_value);
			auto l2 = scalar_loss(reward, target_reward);
			auto l3 = softmax_cross_entropy_with_logits(policy_logits, target_policy);

			std::cout << "Loss_1:\n" << l1 << std::endl;
			std::cout << "Loss_2:\n" << l2 << std::endl;
			std::cout << "Loss_3:\n" << l3 << std::endl;

			auto l = l1 + l2 + l3;
			/*auto l =	scalar_loss(value, target_value) + 
						scalar_loss(reward, target_reward) + 
						softmax_cross_entropy_with_logits(policy_logits, target_policy);*/

			loss += scale_gradient(l, gradient_scale);
		}
		std::cout << "Loss:\n" << loss << std::endl;
	}
	loss.backward();
  	optimizer.step();
	
	return loss.template item<float>();
}

template<class Game>
void MuZero<Game>::train_network()
{
	using namespace std::chrono_literals;

	constexpr size_t num_checkpoints = TRAINING_STEPS / CHEKPOINT_INTERVAL;

	while (!replay_buffer.ready(1))
		std::this_thread::sleep_for(1s);

	for (size_t i = 1; i <= num_checkpoints; ++i) {

		float avg_loss = 0.0;

		for (size_t j = 1; j <= CHEKPOINT_INTERVAL; ++j) {
			auto batch = replay_buffer.sample_batch();
			avg_loss += update_weights(batch);
			std::printf("\r[%2zu/%2zu][%3zu/%3zu]", i, num_checkpoints, j, CHEKPOINT_INTERVAL);
		}
		// Save network, log metrics and test if needed.
		std::printf(" avg_loss: %.4f | games_played: %d -> checkpoint %zu\n",
			avg_loss / CHEKPOINT_INTERVAL,
			games_played.load(),
			i
		);
		torch::save(main_network, out_weights_file);
	}
	train_flag = false;
}

template<class Game>
void MuZero<Game>::selfplay()
{
	// Thread local gradient guard.
	//torch::NoGradGuard no_grad_guard;

	MuZeroNetwork network(net_op);

	network->to(DEVICE);
	network->eval();

	while (train_flag) {
		torch::load(network, out_weights_file);
		auto game = play_game(network);
		replay_buffer.save_game(std::move(game));
		games_played++;
	}
}

template<class Game>
Game MuZero<Game>::play_game(MuZeroNetwork network)
{
	auto game = Game();

	while (!game.terminal() && game.size() < MAX_MOVES) {

		//game.display();

		auto root = std::make_unique<Node>(1.0f);
		auto img = game.make_image();
		auto output = network->initial_inference(img);
		auto legal_actions = game.legal_actions();

		output.policy_logits = filter_policy(output.policy_logits, legal_actions);

		//std::cout << output.policy_logits << std::endl;

		expand(root.get(), game.to_play(), output, legal_actions);	
		add_exploration_noise(root);
		run_mcts(network, root, game.action_history());

		Action action = select_action(root, game.size());

		game.apply(action);
		game.store_search_statistics(root);
	}
	return game;
}

template<class Game>
template<class ActionHistory>
void MuZero<Game>::run_mcts(MuZeroNetwork network, const NodePtr &root, ActionHistory history)
{
	std::vector<Node*> search_path;
	search_path.push_back(root.get());

	for (int i = 0; i < NUM_SIMULATIONS; ++i) {

		auto action = Game::NO_MOVE;
		auto node = root.get();
		int n = 0;

		while (node->expanded()) {
			std::tie(action, node) = select(node, min_max_stats);
			history.add_action(action);
			search_path.push_back(node);
			++n;
		}

		auto parent = search_path.end()[-2];
		auto input_to_dynamics = Game::make_next_state(parent->state, history.last_action());
		auto network_output = network->recurrent_inference(input_to_dynamics);
		
		expand(node, history.to_play(), network_output, Game::ACTION_SPACE);
		backpropagate(search_path, network_output.value.template item<float>(), history.to_play(), min_max_stats);

		history.trim(n);
		search_path.resize(1);
	}
}

}