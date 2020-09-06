#include "neural/mu_zero.h"
#include "games/tictactoe.h"

using namespace mz;

int main(int argc, char **argv) 
{
	NetworkOptions net_op = {
		.in_shape_f = {128, 3, 3},
		.in_shape_g = {129, 3, 3},
		.in_shape_h = {3, 3, 3},
		.filters_f = 128,
		.filters_g = 128,
		.filters_h = 128,
		.num_blocks_f = 4,
		.num_blocks_g = 2,
		.num_blocks_h = 2,
		.reduced_p = 2,
		.reduced_v = 1,
		.reduced_r = 1,
		.downsample = false,
	};

	auto mu = MuZero<TTT>(net_op, false, "checkpoints/tmp_weights.pt", "checkpoints/tmp_weights.pt");

	mu.start_train();
	mu.test_human();
}
