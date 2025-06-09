###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_embedding_bag_sum_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_embedding_bag_sum.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_embedding_bag_sum_op_lib_path))

class CustomEmbeddingBagSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, indices, offset, n_tpc):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_embedding_bag_sum(input_, indices, offset, n_tpc)
        ctx.tensor = tensor
        return tensor

class CustomEmbeddingBagSum(torch.nn.Module):
    def __init__(self):
        super(CustomEmbeddingBagSum, self).__init__()

    def forward(self, input_, indices, offset, n_tpc):
        if len(indices.size()) > 1 or len(offset.size()) > 1:
            print('[Indices] Not support more than 1 dimensions')
            print('[Offset] Not support more than 1 dimensions')
            assert((len(indices.size()) == 1) and (len(offset.size()) == 1))
        
        batchsize = len(offset) - 1
        if batchsize < n_tpc:
            n_tpc = batchsize
            
        return CustomEmbeddingBagSumFunction.apply(input_, indices, offset, n_tpc)

    def extra_repr(self):
        return 'CustomEmbeddingBagSum for float32 only'