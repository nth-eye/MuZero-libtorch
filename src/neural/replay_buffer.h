#include <deque>
#include <random>
#include <chrono>
#include <mutex>
#include <torch/torch.h>

#pragma once

namespace mz {

template<class Game>
struct ReplayBuffer {

	ReplayBuffer() = default;

	bool ready(int n_batches) const { return buffer.size() > BATCH_SIZE * n_batches; }
	void save_game(Game game);
	std::vector<Sample> sample_batch();

	size_t size() { return buffer.size(); }

private:
	Game& sample_game()
	{
		std::uniform_int_distribution<size_t> d(0, buffer.size()-1);
		return buffer[d(rng)];
	}
	size_t sample_position(Game &game) 
	{
		std::uniform_int_distribution<size_t> d(0, game.size()-1);
		return d(rng);
	}

	std::mutex mut;
    std::mt19937 rng{static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
	std::deque<Game> buffer;
};


template<class Game>
void ReplayBuffer<Game>::save_game(Game game)
{
	std::lock_guard lock(mut);
	buffer.push_back(std::move(game));

	if (buffer.size() > WINDOW_SIZE)
		buffer.pop_front();
}

template<class Game>
std::vector<Sample> ReplayBuffer<Game>::sample_batch()
{
	std::vector<Sample> samples;
	//std::lock_guard lock(mut); ???
	for (size_t i = 0; i < BATCH_SIZE; ++i) {

		Game &game = sample_game();
		auto pos = sample_position(game);

		samples.emplace_back(
			game.make_image(pos), 
			game.get_actions(pos), 
			game.make_target(pos)
		);
	}
	return samples;
}

}