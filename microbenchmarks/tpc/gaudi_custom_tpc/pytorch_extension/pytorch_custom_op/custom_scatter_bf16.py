###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_scatter_bf16_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_scatter_bf16.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_scatter_bf16_op_lib_path))

class CustomScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, indices, output_size, n_tpc):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_scatter_bf16(input_, indices, output_size, n_tpc)
        ctx.tensor = tensor
        return tensor

class CustomScatter(torch.nn.Module):
    def __init__(self):
        super(CustomScatter, self).__init__()

    def forward(self, input_, indices, output_size, n_tpc):
        if len(indices.size()) > 1:
            print('[Indices] Not support more than 1 dimensions')
            assert(len(indices.size()) == 1)

        return CustomScatterFunction.apply(input_, indices, output_size, n_tpc)

    def extra_repr(self):
        return 'CustomScatter for bfloat16 only'