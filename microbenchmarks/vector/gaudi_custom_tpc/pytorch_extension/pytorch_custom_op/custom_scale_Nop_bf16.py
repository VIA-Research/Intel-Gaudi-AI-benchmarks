###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_scale_Nop_bf16_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_scale_Nop_bf16.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_scale_Nop_bf16_op_lib_path))

class CustomScaleNOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputA, scalar, N_op):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_scale_Nop_bf16(inputA, scalar, N_op)
        ctx.tensor = tensor
        return tensor

class CustomScaleNOp(torch.nn.Module):
    def __init__(self):
        super(CustomScaleNOp, self).__init__()

    def forward(self, inputA, scalar, N_op):
        return CustomScaleNOpFunction.apply(inputA, scalar, N_op)

    def extra_repr(self):
        return 'CustomScaleNOp for bfloat16 only'