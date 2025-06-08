#!/bin/bash

# Square shaped GEMM
./run_gemm_sqaure.sh power 32 32768 2 bf16

# Irregularly shaped GEMM
./run_gemm_fix_n.sh power 256 32768 2 bf16 1
./run_gemm_fix_n.sh power 256 32768 2 bf16 2
./run_gemm_fix_n.sh power 256 32768 2 bf16 4
./run_gemm_fix_n.sh power 256 32768 2 bf16 8
./run_gemm_fix_n.sh power 256 32768 2 bf16 16
./run_gemm_fix_n.sh power 256 32768 2 bf16 32
./run_gemm_fix_n.sh power 256 32768 2 bf16 64
./run_gemm_fix_n.sh power 256 32768 2 bf16 128
./run_gemm_fix_n.sh power 256 32768 2 bf16 256

# Fix K to 16384 and all-to-all for M and N
./run_gemm_fix_k_all_to_all.sh power 32 16384 2 bf16 16384