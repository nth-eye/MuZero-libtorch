#include "base_game.h"
#include "ttt.h"

#pragma once

namespace mz {

static ActionList action_space()
{
	ActionList list;
	for (size_t i = 0; i < ACTION_SPACE_SIZE; ++i)
		list.push_back(i);
	return list;
}

struct TTT : BaseGame<ttt::State, ttt::Game, TTT> {

	static constexpr Action NO_MOVE = -1;
	static const ActionList ACTION_SPACE;

	static int to_nn_idx(Action action);
	static Action to_action(int index);
	static torch::Tensor make_next_state(torch::Tensor state, Action action);

	// Action handler for tic-tac-toe.
	struct ActionHistory {

		ActionHistory(int turn_) : turn(turn_) {}

		int to_play() { return turn; }
		void trim(int size) { if (size & 1) turn ^= ttt::BOTH; }
		void add_action(Action action_) { action = action_; }
		Action last_action() { return action; }
	private:
		Action action = NO_MOVE;
		int turn;
	};

	void display() const { std::cout << environment.str() << std::endl; }

	ActionHistory action_history() const { return ActionHistory(environment.to_play()); }

	torch::Tensor make_image(int idx) const;
	torch::Tensor make_image() const { return make_image(history.size() - 1); }
};

const ActionList TTT::ACTION_SPACE = action_space();

int TTT::to_nn_idx(Action action)
{
	return action;
}

Action TTT::to_action(int index)
{
	return index;
}

torch::Tensor TTT::make_next_state(torch::Tensor state, Action action) 
{
	// 4-th dim is for batches.
	auto action_plane = torch::zeros({1, 1, 3, 3}, DEVICE);

	action_plane[0][0][action / 3][action % 3] = 1.0f;

	return torch::cat({state, action_plane}, 1);
}

torch::Tensor TTT::make_image(int idx) const 
{
	// 4-th dim is for batches.
	auto board = history[idx];
	auto output = torch::zeros({1, 3, 3, 3});
	auto accessor = output.accessor<float, 4>();

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			switch (board[i*3+j]) {
				case ttt::EMPTY:
					break;
				case ttt::X:
					accessor[0][0][i][j] = 1.0f;
					break;
				case ttt::O:
					accessor[0][1][i][j] = 1.0f;
					break;
				default:
					assert(false);
			}
		}
	}
	if (environment.to_play() == ttt::X) {
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
				accessor[0][2][i][j] = 1.0f;
	}
	return output.to(DEVICE);
}

}