import torch
import argparse
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht

def add(depth, array_size, dtype):
    if dtype == "bfloat16":
        from custom_add_bf16_unroll1 import CustomAddUnroll1
        dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")
        
    addOp = CustomAddUnroll1()

    print(f"Add experiment of array_size: {array_size}, depth: {depth}, dtype: {dtype}")

    warmup = 5
    iteration = 10
    dim0 = depth
    dim1 = array_size // dim0
    for i in range(0, iteration + warmup):
        A = torch.randn((dim1, dim0), dtype=dtype, device='cpu')
        B = torch.randn((dim1, dim0), dtype=dtype, device='cpu')
        A_hpu = A.to("hpu")
        B_hpu = B.to("hpu")
        ht.hpu.synchronize()
        htcore.mark_step()

        C_hpu = addOp(A_hpu, B_hpu)
        ht.hpu.synchronize()
        htcore.mark_step()

def triad(depth, array_size, dtype):
    if dtype == "bfloat16":
        from custom_triad_bf16_unroll1 import CustomTriadUnroll1
        dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")
        
    triadOp = CustomTriadUnroll1()

    print(f"Triad experiment of array_size: {array_size}, depth: {depth}, dtype: {dtype}")

    warmup = 5
    iteration = 10
    dim0 = depth
    dim1 = array_size // dim0
    for i in range(0, iteration + warmup):
        A = torch.randn((dim1, dim0), dtype=dtype, device='cpu')
        B = torch.randn((dim1, dim0), dtype=dtype, device='cpu')
        scalar = 4.0
        A_hpu = A.to("hpu")
        B_hpu = B.to("hpu")
        ht.hpu.synchronize()
        htcore.mark_step()

        C_hpu = triadOp(A_hpu, B_hpu, scalar)
        ht.hpu.synchronize()
        htcore.mark_step()

def scale(depth, array_size, dtype):
    if dtype == "bfloat16":
        from custom_scale_bf16_unroll1 import CustomScaleUnroll1
        dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    scaleOp = CustomScaleUnroll1()

    print(f"Scale experiment of array_size: {array_size}, depth: {depth}, dtype: {dtype}")

    warmup = 5
    iteration = 10
    dim0 = depth 
    dim1 = array_size // dim0
    for i in range(0, iteration + warmup):
        A = torch.randn((dim1, dim0), dtype=dtype, device='cpu')
        scalar = 4.0
        A_hpu = A.to("hpu")
        ht.hpu.synchronize()
        htcore.mark_step()
        
        B_hpu = scaleOp(A_hpu, scalar)
        ht.hpu.synchronize()
        htcore.mark_step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform access granularity test.")
    parser.add_argument("--benchmark", type=str, required=True, help="Type of benchmark: add/scale/triad")
    parser.add_argument("--depth", type=int, required=True, help="Depth size")
    parser.add_argument("--array_size", type=int, required=True, help="Array size")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type: bfloat16")

    args = parser.parse_args()

    assert((args.array_size % (args.depth)) == 0)

    if args.benchmark == "add":
        add(args.depth, args.array_size, args.dtype)
    elif args.benchmark == "scale":
        scale(args.depth, args.array_size, args.dtype)
    elif args.benchmark == "triad":
        triad(args.depth, args.array_size, args.dtype)
