#include <torch/torch.h>

#pragma once

namespace mz {

using Bounds = std::optional<std::pair<float, float>>;

inline constexpr size_t ACTION_SPACE_SIZE = 9;					// Num of all possible actions in game.
inline constexpr size_t NUM_ACTORS = 4;							// Num of threads to fill replay buffer.	
inline constexpr size_t MAX_MOVES = 10;							// Max moves per game in playgame().
inline constexpr size_t NUM_SIMULATIONS = 200;					// Num of expansions in MCTS for single game.
inline constexpr float DISCOUNT = 0.99;							// Future reward discount.
inline constexpr float ALPHA = 0.3;								// Alpha for Dirichlet noise.
inline constexpr float EPSILON = 0.25;							// Epsilon exploration fraction for Dirichlet noise.
inline constexpr int PB_C_BASE = 19'652;						// First coefficient of degree of exploration.
inline constexpr float PB_C_INIT = 1.25;						// Second coefficient of degree of exploration.
inline constexpr Bounds KNOWN_BOUNDS = std::pair(-1.0f, 1.0f);	// Reward min and max bounds.
inline constexpr size_t TRAINING_STEPS = 100; //10'000,			// Number of batches.
inline constexpr size_t CHEKPOINT_INTERVAL = 10; //100,			// Save network weights every N batches.
inline constexpr size_t WINDOW_SIZE = 100'000;					// Max num of games in ReplayBuffer.
inline constexpr size_t BATCH_SIZE = 16; //512,					// Batch size.
inline constexpr size_t NUM_UNROLL_STEPS = 5; //9,				// How far position will be unrolled during training.
inline constexpr size_t TD_STEPS = 9;							// Bootstrapping size to estimate rewards.
inline constexpr float WEIGHT_DECAY = 1e-4;						// Weight decay for optimizer.
inline constexpr float MOMENTUM = 0.9;							// Momentum for optimizer.
inline constexpr float LR_INIT = 0.2;							// Initial learning rate.	
inline constexpr float LR_DECAY_RATE = 0.1;						// Multiply learning rate during decay.
inline constexpr float LR_DECAY_STEPS = 10'000;					// Decay learning rate after N steps.

inline constexpr bool CATEGORICAL_LOSS = false;
inline constexpr int SUPPORT_SIZE = 300;

inline torch::Device DEVICE = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

}