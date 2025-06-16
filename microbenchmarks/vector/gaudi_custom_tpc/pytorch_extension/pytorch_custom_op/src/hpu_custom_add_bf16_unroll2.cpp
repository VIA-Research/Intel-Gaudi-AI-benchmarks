/******************************************************************************
###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>
#include <iostream>

bool register_custom_add_bf16_unroll2() {
    // Registering custom_op::custom_add_bf16_unroll2
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto inputA = inputs[0].toTensor(); // inputA
      std::vector<int64_t> result_sizes = inputA.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::BFloat16, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_add_bf16_unroll2", //schema name
        "custom_add_bf16_unroll2_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::custom_add_bf16_unroll2\n";
    return true;
}


at::Tensor custom_add_bf16_unroll2_execute(
    torch::Tensor inputA,
    torch::Tensor inputB) {
  TORCH_CHECK(inputA.scalar_type() == c10::ScalarType::BFloat16, "InputA matrix expected to be BFloat16 tensor");
  TORCH_CHECK(inputB.scalar_type() == c10::ScalarType::BFloat16, "InputB matrix expected to be BFloat16 tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_add_bf16_unroll2();
  TORCH_CHECK(registered, "custom_add_bf16_unroll2 kernel not registered" );
  std::vector<c10::IValue> inputs{inputA, inputB};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_add_bf16_unroll2");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_add_bf16_unroll2(Tensor self, Tensor other) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_add_bf16_unroll2", custom_add_bf16_unroll2_execute);
}
