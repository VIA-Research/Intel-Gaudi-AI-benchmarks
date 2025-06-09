import torch
import argparse
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht

def add_unroll(unroll, array_size, dtype):
    print(f"Add experiment of array_size: {array_size}, unroll: {unroll}, dtype: {dtype}")
    
    if dtype == "bfloat16":
        dtype = torch.bfloat16
        if unroll == 1:
            from custom_add_bf16_unroll1 import CustomAddUnroll1
            addOp = CustomAddUnroll1()
        elif unroll == 2:
            from custom_add_bf16_unroll2 import CustomAddUnroll2
            addOp = CustomAddUnroll2()
        elif unroll == 3:
            from custom_add_bf16_unroll3 import CustomAddUnroll3
            addOp = CustomAddUnroll3()
        elif unroll == 4:
            from custom_add_bf16 import CustomAdd
            addOp = CustomAdd()
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    warmup = 5
    iteration = 10
    dim0 = 128
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

def triad_unroll(unroll, array_size, dtype):
    print(f"Triad experiment of array_size: {array_size}, unroll: {unroll}, dtype: {dtype}")

    if dtype == "bfloat16":
        dtype = torch.bfloat16
        if unroll == 1:
            from custom_triad_bf16_unroll1 import CustomTriadUnroll1
            triadOp = CustomTriadUnroll1()
        elif unroll == 2:
            from custom_triad_bf16_unroll2 import CustomTriadUnroll2
            triadOp = CustomTriadUnroll2()
        elif unroll == 3:
            from custom_triad_bf16_unroll3 import CustomTriadUnroll3
            triadOp = CustomTriadUnroll3()
        elif unroll == 4:
            from custom_triad_bf16 import CustomTriad
            triadOp = CustomTriad()
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    warmup = 5
    iteration = 10
    dim0 = 128
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

def scale_unroll(unroll, array_size, dtype):
    print(f"Scale experiment of array_size: {array_size}, unroll: {unroll}, dtype: {dtype}")

    if dtype == "bfloat16":
        dtype = torch.bfloat16
        if unroll == 1:
            from custom_scale_bf16_unroll1 import CustomScaleUnroll1
            scaleOp = CustomScaleUnroll1()
        elif unroll == 2:
            from custom_scale_bf16_unroll2 import CustomScaleUnroll2
            scaleOp = CustomScaleUnroll2()
        elif unroll == 3:
            from custom_scale_bf16_unroll3 import CustomScaleUnroll3
            scaleOp = CustomScaleUnroll3()
        elif unroll == 4:
            from custom_scale_bf16 import CustomScale
            scaleOp = CustomScale()
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    warmup = 5
    iteration = 10
    dim0 = 128
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
    parser = argparse.ArgumentParser(description="Perform loop unrolling factor test.")
    parser.add_argument("--benchmark", type=str, required=True, help="Type of benchmark: add/scale/triad")
    parser.add_argument("--unroll", type=int, required=True, help="Unroll size")
    parser.add_argument("--array_size", type=int, required=True, help="Array size")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type: bfloat16")

    args = parser.parse_args()

    assert((args.array_size % (64 * args.unroll)) == 0)

    if args.benchmark == "add":
        add_unroll(args.unroll, args.array_size, args.dtype)
    elif args.benchmark == "scale":
        scale_unroll(args.unroll, args.array_size, args.dtype)
    elif args.benchmark == "triad":
        triad_unroll(args.unroll, args.array_size, args.dtype)
