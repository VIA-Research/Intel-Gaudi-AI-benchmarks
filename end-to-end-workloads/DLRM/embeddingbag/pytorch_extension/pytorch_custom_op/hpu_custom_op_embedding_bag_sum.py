###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import habana_frameworks.torch as ht


def test_custom_embedding_bag_sum_op_function():
    from custom_embedding_bag_sum import CustomEmbeddingBagSum
    print(torch.ops.custom_op.custom_embedding_bag_sum)
    n_tpc = 4
    batchsize = 4
    numlookup = 10
    emb_dim = 65
    input_ = torch.rand((1024, emb_dim), device='cpu')
    indices = torch.randperm(1024, dtype=torch.int, device='cpu')[:batchsize*numlookup]
    offset_list = [numlookup * x for x in range(batchsize + 1)]
    offset = torch.tensor(offset_list, dtype=torch.int, device='cpu')

    input_hpu = input_.to('hpu')
    indices_hpu = indices.to('hpu')
    offset_hpu = offset.to('hpu')
    # output = torch.zeros((2, 4), device='hpu')

    embeddingbag_hpu = CustomEmbeddingBagSum()
    output = embeddingbag_hpu(input_hpu, indices_hpu, offset_hpu, n_tpc)
    print(output)

    ## CPU evaluation
    output_cpu = torch.nn.functional.embedding_bag(indices, input_, offset[:-1], mode='sum')
    print(output_cpu)

    print(torch.allclose(output, output_cpu))




test_custom_embedding_bag_sum_op_function()

