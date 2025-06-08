import torch
import habana_frameworks.torch.core as htcore
import argparse
import os
import habana_frameworks.torch as ht

def main():
	parser = argparse.ArgumentParser(
		description="MxKxN matrix muliplication performance test"
	)
	# dimensions of matrix multiplication
	parser.add_argument("--m", type=int, default=256)
	parser.add_argument("--k", type=int, default=256)
	parser.add_argument("--n", type=int, default=256)

	# datatype
	parser.add_argument("--dtype", type=str, default="fp32")

	args = parser.parse_args()

	m = args.m
	n = args.n
	k = args.k

	if args.dtype == "fp16":
		dtype = torch.float16
	elif args.dtype == "bf16":
		dtype = torch.bfloat16
	elif args.dtype == "fp32":
		dtype = torch.float32
	elif args.dtype == "tf32":
		dtype = torch.float32
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
	else:
		assert False
	
	os.environ["HABANA_PROFILE"] = "1"

	for i in range(1):
		# initialize input matrices
		A = torch.randn(m * k, dtype=dtype).view(m, k)
		B = torch.randn(k * n, dtype=dtype).view(k, n)
		
		A_hpu = A.to("hpu")
		B_hpu = B.to("hpu")

		htcore.mark_step()

		# do matrix multiplication
		C_hpu = torch.mm(A_hpu, B_hpu)

		htcore.mark_step()

		C = C_hpu.to("cpu")	
	
if __name__ == '__main__':
	main()
