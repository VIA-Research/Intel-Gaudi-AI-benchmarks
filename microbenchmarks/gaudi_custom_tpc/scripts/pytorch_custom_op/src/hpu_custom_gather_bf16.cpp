/******************************************************************************
###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>
#include <iostream>

typedef struct sParam{
    int n_tpc;
}Param;

bool register_custom_gather_bf16() {
    // Registering custom_op::custom_gather_bf16
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::SCALAR, 2};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto input = inputs[0].toTensor(); // input
      auto indices = inputs[1].toTensor(); // indices
      auto sizes_input = input.sizes();
      auto sizes_indices = indices.sizes();

      std::vector<int64_t> result_sizes = {sizes_indices[0], sizes_input[1]};
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::BFloat16, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(Param);
      params->n_tpc = inputs[2].toInt(); // bottom
      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_gather_bf16", //schema name
        "custom_gather_bf16_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::custom_gather_bf16\n";
    return true;
}

at::Tensor custom_gather_bf16_execute(
    torch::Tensor input,
    torch::Tensor indices,
    c10::Scalar n_tpc
    ) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::BFloat16, "Input tensor expected to be BFloat16 tensor");
  TORCH_CHECK(indices.scalar_type() == c10::ScalarType::Int, "Input tensor expected to be Int tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_gather_bf16();
  TORCH_CHECK(registered, "custom_gather_bf16 kernel not registered" );
  std::vector<c10::IValue> inputs{input, indices, n_tpc};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_gather_bf16");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_gather_bf16(Tensor t1, Tensor t2, Scalar s) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_gather_bf16", custom_gather_bf16_execute);
}
