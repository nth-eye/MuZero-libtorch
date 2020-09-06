#include <vector>
#include <torch/torch.h>

#pragma once

namespace mz {

using Action = int;
// You can provide other container than vector.
using ActionList = std::vector<Action>;
// Value, reward and policy.
using Target = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
// Observaion, action-list and target.
using Sample = std::tuple<torch::Tensor, ActionList, std::vector<Target>>;

struct Node {

	Node(const float p_) : p(p_) {}

	float get_p() const { return p / (1.0f + static_cast<float>(n)); }
	float get_q() const
	{
		if (!n) return 0.0;
		return w / n;
	}

	bool expanded() const { return children.size() > 0; }
	
	float p = 0.0f;
	float w = 0.0f;
	float r = 0.0f;
	int n = 0;
	int turn = -1;
	torch::Tensor state;

	std::vector<std::pair<Action, std::unique_ptr<Node>>> children;
};
using NodePtr = std::unique_ptr<Node>;

}