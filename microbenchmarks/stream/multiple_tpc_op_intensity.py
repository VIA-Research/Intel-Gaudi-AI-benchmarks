import torch
import argparse
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht

def add_test(unroll, array_size, num_tpc, num_ops, dtype):
    print(f"Experiment of array_size: {array_size}, num_tpc: {num_tpc}, unroll: {unroll}, num_ops: {num_ops}, dtype: {dtype}")
    
    if dtype == torch.bfloat16:
        from custom_add_Nop_bf16 import CustomAddNOp
        addOp = CustomAddNOp()
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    iteration = 10
    warmup = 5
    dim0 = 128
    dim1 = array_size // dim0
    dim2 = num_tpc
    for i in range(0, iteration + warmup):
        A = torch.randn((dim2, dim1, dim0), dtype=dtype, device='cpu')
        B = torch.randn((dim2, dim1, dim0), dtype=dtype, device='cpu')
        A_hpu = A.to("hpu")
        B_hpu = B.to("hpu")
        ht.hpu.synchronize()
        htcore.mark_step()

        C_hpu = addOp(A_hpu, B_hpu, num_ops)
        ht.hpu.synchronize()
        htcore.mark_step()

def scale_test(unroll, array_size, num_tpc, num_ops, dtype):
    print(f"Experiment of array_size: {array_size}, num_tpc: {num_tpc}, unroll: {unroll}, num_ops: {num_ops}, dtype: {dtype}")
    
    if dtype == torch.bfloat16:
        from custom_scale_Nop_bf16 import CustomScaleNOp
        scaleOp = CustomScaleNOp()
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    iteration = 10
    warmup = 5
    dim0 = 128
    dim1 = array_size // dim0
    dim2 = num_tpc
    for i in range(0, iteration + warmup):
        A = torch.randn((dim2, dim1, dim0), dtype=dtype, device='cpu')
        scalar = 4.0
        A_hpu = A.to("hpu")
        ht.hpu.synchronize()
        htcore.mark_step()

        B_hpu = scaleOp(A_hpu, scalar, num_ops)
        ht.hpu.synchronize()
        htcore.mark_step()

def triad_test(unroll, array_size, num_tpc, num_ops, dtype):
    print(f"Experiment of array_size: {array_size}, num_tpc: {num_tpc}, unroll: {unroll}, num_ops: {num_ops}, dtype: {dtype}")

    if dtype == torch.bfloat16:
        from custom_triad_Nop_bf16 import CustomTriadNOp
        triadOp = CustomTriadNOp()
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    iteration = 10
    warmup = 5
    dim0 = 128
    dim1 = array_size // dim0
    dim2 = num_tpc
    for i in range(0, iteration + warmup):
        A = torch.randn((dim2, dim1, dim0), dtype=dtype, device='cpu')
        B = torch.randn((dim2, dim1, dim0), dtype=dtype, device='cpu')
        scalar = 4.0
        A_hpu = A.to("hpu")
        B_hpu = B.to("hpu")
        ht.hpu.synchronize()
        htcore.mark_step()
        
        C_hpu = triadOp(A_hpu, B_hpu, scalar, num_ops)
        ht.hpu.synchronize()
        htcore.mark_step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform unroll test.")
    parser.add_argument("--benchmark", type=str, required=True, help="Unroll size")
    parser.add_argument("--unroll", type=int, required=True, help="Unroll size")
    parser.add_argument("--num_tpc", type=int, required=True, help="Number of TPCs")
    parser.add_argument("--array_size", type=int, required=True, help="Array size")
    parser.add_argument("--dtype", type=str, required=True, help="Data type: bfloat16")
    parser.add_argument("--num_ops", type=int, required=True, help="Number of operations")

    args = parser.parse_args()

    assert((args.array_size % 64) == 0)

    if args.benchmark == "add":
        add_test(args.unroll, args.array_size, args.num_tpc, args.num_ops, dtype)
    elif args.benchmark == "scale":
        scale_test(args.unroll, args.array_size, args.num_tpc, args.num_ops, dtype)
    elif args.benchmark == "triad":
        triad_test(args.unroll, args.array_size, args.num_tpc, args.num_ops, dtype)