import os
import argparse
import pandas as pd
import re

def run():
	parser = argparse.ArgumentParser("Extract the data from Intel Gaudi profiling result csv files")
	parser.add_argument("--dtype", type=str, choices=["fp16", "fp32", "bf16", "tf32"])
	parser.add_argument("--type", type=str, choices=["linear", "power"])
	parser.add_argument("--start", type=int)
	parser.add_argument("--end", type=int)
	parser.add_argument("--stride", type=int)

	args = parser.parse_args()

	dtype = args.dtype
	pattern = args.type
	start = args.start
	end = args.end
	stride = args.stride

	if pattern == "linear":
		assert (end - start) % stride == 0
		values = [start + stride * i for i in range(int((end - start)/stride) + 1)]
	elif pattern == "power":
		values = []
		tmp = start
		while tmp <= end:
			values.append(tmp)
			tmp = tmp * stride
		assert tmp / stride == end

	csv_path = "./trace_files"

	try:
		file_list = os.listdir(csv_path)
	except FileNotFoundError:
		print(f"The directory {csv_path} was not found.")
		return

	print("Square GEMM for %s, M = K = N = %d ~ %d" %(dtype, start, end))
	print()
	print("M, K, N, time (us)")
	k_values = values

	for k in k_values:
		m = k
		n = k

		prefix = "%d_%d_%d_%s" %(m, k, n, dtype)
		matching_files = [file for file in file_list if file.startswith(prefix)]
		assert len(matching_files) == 1, "Duplicated file (Impossible case)"

		file_path = "%s/%s" %(csv_path, matching_files[0])
		try:
			data = pd.read_csv(file_path)
		except FileNotFoundError:
			print(f"The file {file_path} was not found.")
			return
		except pd.errors.EmptyDataError:
			print(f"The file {file_path} is empty.")
			return
		except pd.errors.ParserError:
			print(f"Error parsing the CSV file {file_path}.")
			return

		idx_first = data["Start time of node"].idxmin()
		idx_last = data["Start time of node"].idxmax()
  
		start = data.loc[idx_first]["Start time of node"]
		end = data.loc[idx_last]["End time of node"]
  
		time = end - start
  
		print("%d, %d, %d, %f" %(m, k, n, time))

if __name__ == "__main__":
	run()

