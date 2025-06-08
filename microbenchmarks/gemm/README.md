# Intel Gaudi GEMM Benchmark

This repository provides scripts and helper tools to measure and analyze the performance of General Matrix–Matrix Multiplication (GEMM) on Habana Gaudi NPUs using PyTorch and Habana frameworks.

---
###  Prerequisites
Install Python dependencies:
```bash
pip install torch pandas
```

---

### Repository Structure
```bash
.
├── gemm.py
├── run_entire_experiments.sh
├── run_gemm_sqaure.sh
├── run_gemm_fix_n.sh
├── run_gemm_fix_k_all_to_all.sh
├── parser_square_time.py
├── parser_fixed_n_time.py
└── trace_files/  # raw CSV outputs from Habana profiling
```

- **gemm.py**  
  - Launches a single M×K×N matrix-multiply on the Gaudi device.  
  - Configurable data type (`fp16`, `fp32`, `bf16`, `tf32`) and dimensions.  
  - Habana profiler result files will be generated automatically if the profiler is enabled before running this Python script

- **run_gemm_sqaure.sh**  
  - Sweeps through square GEMM sizes (M=K=N) across the specified range and executes a GEMM for each configuration.  
    - If "linear" is selected, sweep from "start" to "end", adding "stride" at each configuration.
    - If "power" is selected, sweep from "start" to "end", multiplying by "stride" at each configuration.
  - Generates one CSV per configuration in `trace_files/`.
  - Usage:  
    ```bash
    ./run_gemm_sqaure.sh <linear|power> <start> <end> <stride> <dtype>
    ```

- **run_gemm_fix_n.sh**  
  - Fixes N constant, sweeps M=K.  
  - Usage:  
    ```bash
    ./run_gemm_fix_n.sh <linear|power> <start> <end> <stride> <dtype> <fixed-N>
    ```

- **run_gemm_fix_k_all_to_all.sh**  
  - Fixes K constant, sweeps M and N independently.  
  - Usage:  
    ```bash
    ./run_gemm_fix_k_all_to_all.sh <linear|power> <start> <end> <stride> <dtype> <fixed-K>
    ```

- **run_entire_experiments.sh**  
  - Wrapper that runs the square, fixed-N, and fixed-K suites in sequence.

- **parser_square_time.py**  
  - Reads `trace_files/*.csv` from square GEMM runs and prints `M, K, N, elapsed_time` to stdout.

- **parser_fixed_n_time.py**  
  - Reads `trace_files/*.csv` from fixed-N (or fixed-K) runs and prints `M, K, N, elapsed_time`.

<!-- ## Prerequisites

- **Python 3.7+**  
- **PyTorch** built with Habana support  
- **habana_frameworks.torch** (and related Habana packages)  
- **pandas** (for parsing scripts)  
- A working **Habana Gaudi** runtime and driver installation -->

