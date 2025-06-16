import torch
import argparse
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch as ht

def add_test(array_size, num_tpc, dtype):
    print(f"Add experiment of array_size: {array_size}, num_tpc: {num_tpc}, dtype: {dtype}")
    if dtype == "bfloat16":
        dtype = torch.bfloat16
        from custom_add_bf16 import CustomAdd
        addOp = CustomAdd()
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

        C_hpu = addOp(A_hpu, B_hpu)
        ht.hpu.synchronize()
        htcore.mark_step()


def scale_test(array_size, num_tpc, dtype):
    print(f"Scale experiment of array_size: {array_size}, num_tpc: {num_tpc}, dtype: {dtype}")
    if dtype == "bfloat16":
        dtype = torch.bfloat16
        from custom_scale_bf16 import CustomScale
        scaleOp = CustomScale()
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

        B_hpu = scaleOp(A_hpu, scalar)
        ht.hpu.synchronize()
        htcore.mark_step()

def triad_test(array_size, num_tpc, dtype):
    print(f"Triad experiment of array_size: {array_size}, num_tpc: {num_tpc}, dtype: {dtype}")

    if dtype == "bfloat16":
        dtype = torch.bfloat16
        from custom_triad_bf16 import CustomTriad
        triadOp = CustomTriad()
    else:
        raise ValueError("Unsupported data type. Only bfloat16 is supported.")

    warmup = 5
    iteration = 10
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

        C_hpu = triadOp(A_hpu, B_hpu, scalar)
        ht.hpu.synchronize()
        htcore.mark_step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform multiple TPCs test.")
    parser.add_argument("--benchmark", type=str, required=True, help="Type of benchmark: add/scale/triad")
    parser.add_argument("--num_tpc", type=int, required=True, help="Number of TPCs")
    parser.add_argument("--array_size", type=int, required=True, help="Array size")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type: bfloat16")

    args = parser.parse_args()

    assert((args.array_size % 64) == 0)

    if args.benchmark == "add":
        add_test(args.array_size, args.num_tpc, dtype)
    elif args.benchmark == "scale":
        scale_test(args.array_size, args.num_tpc, dtype)
    elif args.benchmark == "triad":
        triad_test(args.array_size, args.num_tpc, dtype)