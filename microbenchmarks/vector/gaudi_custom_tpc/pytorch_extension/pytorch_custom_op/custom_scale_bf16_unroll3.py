###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_scale_bf16_unroll3_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_scale_bf16_unroll3.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_scale_bf16_unroll3_op_lib_path))

class CustomScaleUnroll3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputA, scalar):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_scale_bf16_unroll3(inputA, scalar)
        ctx.tensor = tensor
        return tensor

class CustomScaleUnroll3(torch.nn.Module):
    def __init__(self):
        super(CustomScaleUnroll3, self).__init__()

    def forward(self, inputA, scalar):
        return CustomScaleUnroll3Function.apply(inputA, scalar)

    def extra_repr(self):
        return 'CustomScaleUnroll3 for bfloat16 only'