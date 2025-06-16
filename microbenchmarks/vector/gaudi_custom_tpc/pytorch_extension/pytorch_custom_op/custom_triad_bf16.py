###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_triad_bf16_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_triad_bf16.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_triad_bf16_op_lib_path))

class CustomTriadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputA, inputB, scalar):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_triad_bf16(inputA, inputB, scalar)
        ctx.tensor = tensor
        return tensor

class CustomTriad(torch.nn.Module):
    def __init__(self):
        super(CustomTriad, self).__init__()

    def forward(self, inputA, inputB, scalar):
        return CustomTriadFunction.apply(inputA, inputB, scalar)

    def extra_repr(self):
        return 'CustomTriad for bfloat16 only'