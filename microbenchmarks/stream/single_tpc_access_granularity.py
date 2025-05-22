import os
import sys
import torch
import time
import argparse
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht

def add(unroll, array_size, dtype):
    if dtype == "bfloat16":
        from custom_add_bf16_unroll1 import CustomAddUnroll1
        data_size = 2
        dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")
        
    addOp = CustomAddUnroll1()

    print(f"Add experiment of array_size: {array_size}, unroll: {unroll}, dtype: {dtype}")
    print(f"Memory per array: {array_size * data_size / (1024 * 1024)} MB")

    warmup = 5
    iteration = 10
    dim0 = unroll
    dim1 = array_size // unroll
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

def triad(unroll, array_size, dtype):
    if dtype == "bfloat16":
        from custom_triad_bf16_unroll1 import CustomTriadUnroll1
        data_size = 2
        dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")
        
    triadOp = CustomTriadUnroll1()

    print(f"Triad experiment of array_size: {array_size}, unroll: {unroll}, dtype: {dtype}")
    print(f"Memory per array: {array_size * data_size / (1024 * 1024)} MB")

    warmup = 5
    iteration = 10
    dim0 = unroll
    dim1 = array_size // unroll
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

def scale(unroll, array_size, dtype):
    if dtype == "bfloat16":
        from custom_scale_bf16_unroll1 import CustomScaleUnroll1
        data_size = 2
        dtype = torch.bfloat16
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    scaleOp = CustomScaleUnroll1()

    print(f"Scale experiment of array_size: {array_size}, unroll: {unroll}, dtype: {dtype}")
    print(f"Memory per array: {array_size * data_size / (1024 * 1024)} MB")

    warmup = 5
    iteration = 10
    dim0 = unroll 
    dim1 = array_size // unroll
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
    parser = argparse.ArgumentParser(description="Perform unroll test.")
    parser.add_argument("--benchmark", type=str, required=True, help="Type of benchmark")
    parser.add_argument("--unroll", type=int, required=True, help="Unroll size")
    parser.add_argument("--array_size", type=int, required=True, help="Array size")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type: bfloat16")

    args = parser.parse_args()

    assert((args.array_size % (args.unroll)) == 0)

    if args.benchmark == "add":
        add(args.unroll, args.array_size, args.dtype)
    elif args.benchmark == "scale":
        scale(args.unroll, args.array_size, args.dtype)
    elif args.benchmark == "triad":
        triad(args.unroll, args.array_size, args.dtype)
