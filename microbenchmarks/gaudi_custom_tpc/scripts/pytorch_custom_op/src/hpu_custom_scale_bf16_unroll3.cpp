/******************************************************************************
###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>
#include <iostream>

typedef struct Param{
    float scalar;
} scaleParam;

bool register_custom_scale_bf16_unroll3() {
    // Registering custom_op::custom_scale_bf16_unroll3
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::SCALAR, 1};

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

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(scaleParam);
      params->scalar = inputs[1].toDouble(); // bottom
      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_scale_bf16_unroll3", //schema name
        "custom_scale_bf16_unroll3_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::custom_scale_bf16_unroll3\n";
    return true;
}

at::Tensor custom_scale_bf16_unroll3_execute(
    torch::Tensor inputA,
    c10::Scalar scalar) {
  TORCH_CHECK(inputA.scalar_type() == c10::ScalarType::BFloat16, "InputA matrix expected to be BFloat16 tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_scale_bf16_unroll3();
  TORCH_CHECK(registered, "custom_scale_bf16_unroll3 kernel not registered" );
  std::vector<c10::IValue> inputs{inputA, scalar};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_scale_bf16_unroll3");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_scale_bf16_unroll3(Tensor self, Scalar side) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_scale_bf16_unroll3", custom_scale_bf16_unroll3_execute);
}
