# Intel Gaudi TPC Microbenchmarks

This repository provides a collection of microbenchmarks for evaluating the performance characteristics of custom TPC (Tensor Processing Core) kernels on the Intel Gaudi-2 AI processor. The benchmarks measure the efficiency of various operations such as `add`, `scale`, and `triad` under different configurations of access granularity, TPC utilization, operation intensity, and loop unrolling.

## Repository Structure

```
.
├── gather_scatter/
│   └── tpc_emb_gather_scatter.py         # Embedding gather/scatter microbenchmarks
├── stream/
│   ├── multiple_tpc_num_tpc_units.py     # Multi-TPC benchmark for varying TPC unit count
│   ├── multiple_tpc_op_intensity.py      # Multi-TPC benchmark for operation intensity
│   ├── single_tpc_access_granularity.py  # Single-TPC benchmark for access granularity
│   └── single_tpc_unrolling_factor.py    # Single-TPC benchmark for loop unrolling
├── gaudi_custom_tpc/                     # Custom TPC kernel implementations (must be installed prior to benchmarking)
```

## Prerequisites

- Intel Gaudi-2 System
- Habana® SynapseAI™ stack
- PyTorch with Habana integration (`habana_frameworks.torch`)

> **Note:** Before running any benchmark, you must first build and install the custom TPC kernels provided in the `gaudi_custom_tpc/` directory.

## Benchmarks

### 1. Single-TPC: Access Granularity
**Location**: `stream/single_tpc_access_granularity.py`  
Measures performance based on depth-wise data access patterns.

```bash
python single_tpc_access_granularity.py \
  --benchmark triad \
  --depth 128 \
  --array_size 25165824 \
  --dtype bfloat16
```

### 2. Single-TPC: Loop Unrolling Factor
**Location**: `stream/single_tpc_unrolling_factor.py`  
Compares performance impact of different loop unrolling factors.

```bash
python single_tpc_unrolling_factor.py \
  --benchmark add \
  --unroll 4 \
  --array_size 25165824 \
  --dtype bfloat16
```

### 3. Multi-TPC: Varying Number of TPC Units
**Location**: `stream/multiple_tpc_num_tpc_units.py`  
Evaluates how performance scales with the number of active TPC units.

```bash
python multiple_tpc_num_tpc_units.py \
  --benchmark add \
  --array_size 25165824 \
  --num_tpc 8 \
  --dtype bfloat16
```

### 4. Multi-TPC: Operation Intensity
**Location**: `stream/multiple_tpc_op_intensity.py`  
Tests how varying the number of FLOPs per TPC impacts performance.

```bash
python multiple_tpc_op_intensity.py \
  --benchmark scale \
  --array_size 25165824 \
  --num_tpc 24 \
  --num_ops 1024 \
  --dtype bfloat16
```

### 5. Embedding Gather/Scatter
**Location**: `gather_scatter/tpc_emb_gather_scatter.py`  
Tests gather and scatter operations on embedding tables using custom TPC kernels.

```bash
python tpc_emb_gather_scatter.py \
  --benchmark gather \
  --emb_dim 128 \
  --num_emb 4194304 \
  --num_tpc 24 \
  --num_updates 16384 \
  --dtype bfloat16
```

## Notes

- Custom kernels must be compiled and accessible to Python before use.
