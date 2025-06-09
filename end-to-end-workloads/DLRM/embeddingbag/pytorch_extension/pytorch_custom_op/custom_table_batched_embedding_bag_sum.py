###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_table_batched_embedding_bag_sum_op_lib_path = "./build/lib.linux-x86_64-cpython-310/hpu_custom_table_batched_embedding_bag_sum.cpython-310-x86_64-linux-gnu.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_table_batched_embedding_bag_sum_op_lib_path))

class CustomTableBatchedEmbeddingBagSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, weight_width_offset, indices, offset, n_tpc, num_table):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_table_batched_embedding_bag_sum(weight, weight_width_offset, indices, offset, n_tpc, num_table)
        ctx.tensor = tensor
        return tensor

class CustomTableBatchedEmbeddingBagSum(torch.nn.Module):
    def __init__(self):
        super(CustomTableBatchedEmbeddingBagSum, self).__init__()

    def forward(self, weight, weight_width_offset, indices, offset, n_tpc, num_table):
        if len(indices.size()) > 1 or len(offset.size()) > 1:
            print('[Indices] Not support more than 1 dimensions')
            print('[Offset] Not support more than 1 dimensions')
            assert((len(indices.size()) == 1) and (len(offset.size()) == 1))
        batchsize = len(offset) - 1
        assert(batchsize >= num_table)
        if batchsize < n_tpc:
            n_tpc = batchsize
        batchsize_per_table = int(batchsize / num_table)
        
        output = CustomTableBatchedEmbeddingBagSumFunction.apply(weight, weight_width_offset, indices, offset, n_tpc, num_table)
        return output.reshape(batchsize_per_table, output.shape[1] * num_table)

    def extra_repr(self):
        return 'CustomTableBatchedEmbeddingBagSum for float32 only'