#include "misc.h"

#pragma once

namespace mz {

using namespace torch;

struct ResBlockImpl : nn::Module {

	ResBlockImpl(int filters) :
		conv_1(nn::Conv2dOptions(filters, filters, 3).padding(1)),
		conv_2(nn::Conv2dOptions(filters, filters, 3).padding(1)),
		batch_norm_1(filters),
		batch_norm_2(filters)
	{
		register_module("conv_1", conv_1);
		register_module("conv_2", conv_2);
		register_module("batch_norm_1", batch_norm_1);
		register_module("batch_norm_2", batch_norm_2);
	}

	Tensor forward(Tensor x) 
	{
		auto identity = x.clone();

		x = batch_norm_1(conv_1(x));
		x = torch::relu(x);
		x = batch_norm_2(conv_2(x));

		x += identity;
		x = torch::relu(x);

		return x;
	}
private:
	nn::Conv2d conv_1 = nullptr;
	nn::Conv2d conv_2 = nullptr;
	nn::BatchNorm2d batch_norm_1 = nullptr;
	nn::BatchNorm2d batch_norm_2 = nullptr;
};
TORCH_MODULE(ResBlock);

struct ReduceAndFlatImpl : nn::Module {

	ReduceAndFlatImpl(const int in_shape[3], int reduced, int out_size) :
		conv(nn::Conv2dOptions(in_shape[0], reduced, 1)),
		batch_norm(reduced),
		fc(in_shape[1] * in_shape[2] * reduced, out_size)
	{
		register_module("conv", conv);
		register_module("batch_norm", batch_norm);
		register_module("fc", fc);
	}

	Tensor forward(Tensor x)
	{
		x = torch::relu(batch_norm(conv(x)));
		x = x.view({x.size(0), -1});
		x = fc(x);

		return x;
	}
private:
	nn::Conv2d conv = nullptr;
	nn::BatchNorm2d batch_norm = nullptr;
	nn::Linear fc = nullptr;
};
TORCH_MODULE(ReduceAndFlat);

struct PredictionImpl : nn::Module {

	PredictionImpl(const int in_shape[3], int filters, int num_blocks, int reduced_p, int reduced_v) :
		p_head(in_shape, reduced_p, ACTION_SPACE_SIZE)
	{
		if (CATEGORICAL_LOSS)
			v_head = ReduceAndFlat(in_shape, reduced_v, SUPPORT_SIZE);
		else
			v_head = ReduceAndFlat(in_shape, reduced_v, 1);

		register_module("p_head", p_head);
		register_module("v_head", v_head);

		for (int i = 1; i <= num_blocks; ++i) {
			auto block = res_blocks.emplace_back(filters);
			register_module("res_block_"+std::to_string(i), block);
		}
	}

	std::pair<Tensor, Tensor> forward(Tensor x)
	{
		for (auto &block : res_blocks)
			x = block(x);

		auto policy = p_head->forward(x);
		auto value = v_head->forward(x);

		if constexpr (CATEGORICAL_LOSS)
			value = support_to_scalar(value);
		else
			value = tanh(value);

		return {policy, value};
	}
private:
	ReduceAndFlat p_head = nullptr;
	ReduceAndFlat v_head = nullptr;;
	std::vector<ResBlock> res_blocks;
};
TORCH_MODULE(Prediction);

struct DynamicsImpl : nn::Module {

	DynamicsImpl(const int in_shape[3], int filters, int num_blocks, int reduced) :
		input_block(
			nn::Conv2d(in_shape[0], filters, 1),
			nn::BatchNorm2d(filters), 
			nn::ReLU())
	{
		int shape[] = {filters, in_shape[1], in_shape[2]};

		if constexpr (CATEGORICAL_LOSS)
			reward_block = ReduceAndFlat(shape, reduced, SUPPORT_SIZE);
		else
			reward_block = ReduceAndFlat(shape, reduced, 1);

		register_module("input_block", input_block);
		register_module("reward_block", reward_block);

		for (int i = 1; i <= num_blocks; ++i) {
			auto block = res_blocks.emplace_back(filters);
			register_module("res_block_"+std::to_string(i), block);
		}
	}

	std::pair<Tensor, Tensor> forward(Tensor x)
	{
		x = input_block->forward(x);
		for (auto &block : res_blocks)
			x = block(x);

		auto state = x;
		auto reward = reward_block->forward(x);

		if constexpr (CATEGORICAL_LOSS)
			reward = support_to_scalar(reward);
		else
			reward = tanh(reward);

		return {state, reward};
	}
private:
	nn::Sequential input_block;
	ReduceAndFlat reward_block = nullptr;
	std::vector<ResBlock> res_blocks;
};
TORCH_MODULE(Dynamics);

struct RepresentationImpl : nn::Module {

	RepresentationImpl(const int in_shape[3], int filters, int num_blocks, bool downsample)
	{
		if (downsample) {
			input_block = nn::Sequential(
				nn::Conv2d(nn::Conv2dOptions(in_shape[0], filters/2, 3).stride(2).padding(1)),
				ResBlock(filters),
				ResBlock(filters),
				nn::Conv2d(nn::Conv2dOptions(filters/2, filters, 3).stride(2).padding(1)),
				ResBlock(filters),
				ResBlock(filters),
				ResBlock(filters),
				nn::AvgPool2d(nn::AvgPool2dOptions(3).stride({2, 2})),
				ResBlock(filters),
				ResBlock(filters),
				ResBlock(filters),
				nn::AvgPool2d(nn::AvgPool2dOptions(3).stride({2, 2}))
			);
		} else {
			input_block = nn::Sequential(
				nn::Conv2d(nn::Conv2dOptions(in_shape[0], filters, 3).padding(1)),
				nn::BatchNorm2d(filters),
				nn::ReLU()
			);
		}
		register_module("input_block", input_block);

		for (int i = 1; i <= num_blocks; ++i) {
			auto block = res_blocks.emplace_back(filters);
			register_module("res_block_"+std::to_string(i), block);
		}
	}

	Tensor forward(Tensor x)
	{
		x = input_block->forward(x);
		for (auto &block : res_blocks)
			x = block(x);
		return x;
	}
private:
	nn::Sequential input_block;
	std::vector<ResBlock> res_blocks;
};
TORCH_MODULE(Representation);

struct MuZeroNetworkImpl : nn::Module {

	MuZeroNetworkImpl(const NetworkOptions &o) :
		f(o.in_shape_f, o.filters_f, o.num_blocks_f, o.reduced_p, o.reduced_v),
		g(o.in_shape_g, o.filters_g, o.num_blocks_g, o.reduced_v),
		h(o.in_shape_h, o.filters_h, o.num_blocks_h, o.downsample)
	{
		register_module("f", f);
		register_module("g", g);
		register_module("h", h);
	}

	NetworkOutput initial_inference(Tensor observation)
	{
		auto hidden_state = h(observation);
		auto [policy_logits, value] = f(hidden_state);

		return {value, torch::zeros(1, DEVICE), policy_logits, hidden_state};
	}
	NetworkOutput recurrent_inference(Tensor input_state)
	{
		auto [output_state, reward] = g(input_state);
        auto [policy_logits, value] = f(output_state);

        return {value, reward, policy_logits, output_state};
	}
private:
	Prediction f;
	Dynamics g;
	Representation h;
};
TORCH_MODULE(MuZeroNetwork);

}