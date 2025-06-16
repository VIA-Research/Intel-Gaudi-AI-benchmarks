###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_scale_bf16_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_scale_bf16.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_scale_bf16_op_lib_path))

class CustomScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputA, scalar):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_scale_bf16(inputA, scalar)
        ctx.tensor = tensor
        return tensor

class CustomScale(torch.nn.Module):
    def __init__(self):
        super(CustomScale, self).__init__()

    def forward(self, inputA, scalar):
        return CustomScaleFunction.apply(inputA, scalar)

    def extra_repr(self):
        return 'CustomScale for bfloat16 only'