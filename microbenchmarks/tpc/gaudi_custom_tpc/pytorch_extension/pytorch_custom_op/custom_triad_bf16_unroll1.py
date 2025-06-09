###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_triad_bf16_unroll1_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_triad_bf16_unroll1.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_triad_bf16_unroll1_op_lib_path))

class CustomTriadUnroll1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputA, inputB, scalar):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_triad_bf16_unroll1(inputA, inputB, scalar)
        ctx.tensor = tensor
        return tensor

class CustomTriadUnroll1(torch.nn.Module):
    def __init__(self):
        super(CustomTriadUnroll1, self).__init__()

    def forward(self, inputA, inputB, scalar):
        return CustomTriadUnroll1Function.apply(inputA, inputB, scalar)

    def extra_repr(self):
        return 'CustomTriadUnroll1 for bfloat16 only'