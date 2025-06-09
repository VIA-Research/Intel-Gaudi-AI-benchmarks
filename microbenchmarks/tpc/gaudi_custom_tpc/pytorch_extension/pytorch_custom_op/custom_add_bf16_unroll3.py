###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_add_bf16_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_add_bf16_unroll3.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_add_bf16_op_lib_path))

class CustomAddUnroll3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputA, inputB):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_add_bf16_unroll3(inputA, inputB)
        ctx.tensor = tensor
        return tensor

class CustomAddUnroll3(torch.nn.Module):
    def __init__(self):
        super(CustomAddUnroll3, self).__init__()

    def forward(self, inputA, inputB):
        return CustomAddUnroll3Function.apply(inputA, inputB)

    def extra_repr(self):
        return 'CustomAddUnroll3 for bfloat16 only'
