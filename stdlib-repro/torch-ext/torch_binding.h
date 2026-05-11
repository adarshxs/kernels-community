#pragma once

#include <torch/torch.h>

#include <string>

std::string relu_and_probe(
    torch::Tensor& out,
    torch::Tensor const& input,
    std::string const& code,
    int64_t iters);
