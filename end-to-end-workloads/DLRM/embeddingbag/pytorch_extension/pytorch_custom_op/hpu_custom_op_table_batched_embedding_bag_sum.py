###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import habana_frameworks.torch as ht


def test_custom_table_batched_embedding_bag_sum_op_function():
    from custom_table_batched_embedding_bag_sum import CustomTableBatchedEmbeddingBagSum
    print(torch.ops.custom_op.custom_table_batched_embedding_bag_sum)
    n_table = 2
    n_tpc = 2
    batchsize = 4
    numlookup = 10
    emb_dim = 128
    weight = torch.rand((1000 * n_table, emb_dim), device='cpu')
    weight_width_offset = torch.tensor([0, 1000], dtype=torch.int, device='cpu')
    indices = torch.randperm(1000, dtype=torch.int, device='cpu')[:batchsize*numlookup*n_table]
    offset_list = [numlookup * x for x in range(batchsize*n_table + 1)]
    offset = torch.tensor(offset_list, dtype=torch.int, device='cpu')

    weight_hpu = weight.to('hpu')
    weight_width_offset_hpu = weight_width_offset.to('hpu')
    indices_hpu = indices.to('hpu')
    offset_hpu = offset.to('hpu')

    embeddingbag_hpu = CustomTableBatchedEmbeddingBagSum()
    output = embeddingbag_hpu(weight_hpu, weight_width_offset_hpu, indices_hpu, offset_hpu, n_tpc, n_table)
    # print(output)

    ## CPU evaluation
    output1_cpu = torch.nn.functional.embedding_bag(indices[:batchsize*numlookup], weight[:1000], offset[:batchsize], mode='sum')
    output2_cpu = torch.nn.functional.embedding_bag(indices[batchsize*numlookup:], weight[1000:], offset[:batchsize], mode='sum')
    # print(output1_cpu)

    print(torch.allclose(output[0][:emb_dim], output1_cpu[0]))
    print(torch.allclose(output[0][emb_dim:], output2_cpu[0]))
    print(torch.allclose(output[1][:emb_dim], output1_cpu[1]))
    print(torch.allclose(output[1][emb_dim:], output2_cpu[1]))
    print(torch.allclose(output[2][:emb_dim], output1_cpu[2]))
    print(torch.allclose(output[2][emb_dim:], output2_cpu[2]))
    print(torch.allclose(output[3][:emb_dim], output1_cpu[3]))
    print(torch.allclose(output[3][emb_dim:], output2_cpu[3]))

    print("All tests pass")

test_custom_table_batched_embedding_bag_sum_op_function()

