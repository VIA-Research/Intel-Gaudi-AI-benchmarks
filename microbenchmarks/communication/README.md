# HCCL Benchmark Runner

This repository provides an automated benchmarking script for Habana Collective Communication Library (HCCL), following the guidelines and demo implementation provided by the official Habana GitHub repository: [https://github.com/HabanaAI/hccl_demo](https://github.com/HabanaAI/hccl_demo).

## Overview

The benchmarking is conducted using the `run_hccl_demo.py` script from Habana’s `hccl_demo`, and evaluates the performance of key collective operations across different ranks.

The included script `run_hccl_test.sh` automates multi-rank performance testing over a range of message sizes and collective communication patterns, storing results in both CSV and log formats.

## Supported Collective Operations

The script tests the following HCCL primitives:

- `all_reduce`
- `all_gather`
- `all2all`
- `broadcast`
- `reduce`
- `reduce_scatter`

## Benchmark Parameters

Each test is run with:
- `nranks`: 2 to 8
- `data_type`: `bfloat16`
- `message size range`: 1KB to 1GB (inclusive)
- `loop`: 20 iterations
- `output`: per-operation logs and CSVs stored in the `result/` directory

## How to Run

Ensure that the dependencies and environment setup required by `hccl_demo` are satisfied. Then execute:

```bash
bash run_hccl_test.sh
```

This will:
- Create a `result/` directory
- Iterate through collective operations and rank configurations
- Save logs as `result/<operation>_<nranks>.log`
- Save CSVs as `result/<operation>_result_<nranks>.csv`

## Note

This benchmarking approach is aligned with the official Habana HCCL demo repository:  
👉 [https://github.com/HabanaAI/hccl_demo](https://github.com/HabanaAI/hccl_demo)

Please refer to that repository for additional information about HCCL setup, supported hardware, and advanced options.
