import torch
import argparse
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht

def emb_gather_test(emb_dim, num_emb, num_updates, num_tpc, dtype):
    print(f"Gather experiment of emb_dim: {emb_dim}, num_emb: {num_emb}, num_updates: {num_updates}, num_tpc: {num_tpc}, dtype: {dtype}")
    if dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    from custom_gather_bf16 import CustomGather
    Op = CustomGather()
    dim0 = emb_dim
    dim1 = num_emb
    size_input = (dim1, dim0)

    warmup = 5
    iteration = 10
    for i in range(0, iteration + warmup):
        input_ = torch.arange(dim1*dim0, dtype=dtype, device='cpu').view(size_input)
        indice = torch.randperm(num_emb, dtype=torch.int, device='cpu')[:num_updates]
        input_hpu = input_.to('hpu')
        indice_hpu = indice.to('hpu')
        ht.hpu.synchronize()
        htcore.mark_step()

        output_hpu = Op(input_hpu, indice_hpu, num_tpc)
        ht.hpu.synchronize()
        htcore.mark_step()

def emb_scatter_test(emb_dim, num_emb, num_updates, num_tpc, dtype):
    print(f"Scatter experiment of emb_dim: {emb_dim}, num_emb: {num_emb}, num_updates: {num_updates}, num_tpc: {num_tpc}, dtype: {dtype}")
    if dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    from custom_scatter_bf16 import CustomScatter
    Op = CustomScatter()
    dim0 = emb_dim
    dim1 = num_emb
    size_input = (num_updates, dim0)
    size_output = (num_emb, dim0)

    warmup = 5
    iteration = 10
    for i in range(0, iteration + warmup):
        output = torch.zeros(size_output, dtype=dtype, device='cpu')
        input_ = torch.arange(num_updates*dim0, dtype=dtype, device='cpu').view(size_input)
        indice = torch.randperm(num_emb, dtype=torch.int, device='cpu')[:num_updates]
        input_hpu = input_.to('hpu')
        indice_hpu = indice.to('hpu')
        output_hpu = output.to('hpu')
        ht.hpu.synchronize()
        htcore.mark_step()

        output_hpu = Op(input_hpu, indice_hpu, num_emb, num_tpc)
        ht.hpu.synchronize()
        htcore.mark_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform stride test.")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark: gather/scatter")
    parser.add_argument("--emb_dim", type=int, required=True, help="Embedding dimension")
    parser.add_argument("--num_emb", type=int, required=True, help="Number of embedding")
    parser.add_argument("--num_tpc", type=int, required=True, help="Number of TPC units")
    parser.add_argument("--num_updates", type=int, required=True, help="Number of gather/scatter")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type: bfloat16")

    args = parser.parse_args()

    if args.benchmark == "gather":
        emb_gather_test(args.emb_dim, args.num_emb, args.num_updates, args.num_tpc, args.dtype)
    elif args.benchmark == "scatter":
        emb_scatter_test(args.emb_dim, args.num_emb, args.num_updates, args.num_tpc, args.dtype)