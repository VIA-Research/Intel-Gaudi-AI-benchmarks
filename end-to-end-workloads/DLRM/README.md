# DLRM DCN Benchmark on Habana Gaudi

This repository provides an end-to-end pipeline for benchmarking Deep Learning Recommendation Models (DLRM) with Deep & Cross Networks (DCN) on Habana Gaudi processors. It includes synthetic data generation, PyTorch-based model implementation with embeddingbag, and benchmarking scripts for evaluating throughput across different batch sizes and embedding dimensions.

## Project Structure

```
.
├── bench/
│   └── dlrm_dcn_s_benchmark_hpu.sh   # Benchmark automation script for RM1 and RM2 settings
├── embeddingbag/                     # Custom TPC implementation of embeddingbag for Gaudi support
├── dlrm_dcn_s_pytorch_hpu.py         # PyTorch implementation of DLRM-DCN with Gaudi support
├── dlrm_data_pytorch.py              # Dataset loader and random/synthetic data generators
├── make_random_data_hpu.py           # Script to generate and save random test datasets for inference
├── dataset/                          # Folder where generated test data will be saved
└── logs/                             # Output log files for benchmarking runs
```

## Prerequisites

- PyTorch with Habana Gaudi HPU backend
- `habana_frameworks` and `hpex` libraries
- Gaudi hardware (e.g., Gaudi2) and SynapseAI installed

## Custom TPC Kernel Setup

Before running any scripts, make sure to set up the custom TPC kernel for DLRM's `EmbeddingBag`.
This kernel is required for execution on Gaudi and is located in the `embeddingbag/` directory.

Follow the setup instructions provided in `embeddingbag/` to compile and install the necessary TPC operator.

## Usage

### 1. Random Data Generation

To generate input datasets for inference benchmarking:

```bash
python make_random_data_hpu.py
```

This creates `.pt` files under the `dataset/` directory for two variants:

- `DLRM_DCN_RM1`: 10 tables, 10 lookups
- `DLRM_DCN_RM2`: 20 tables, 100 lookups

Each file contains 101 batches for a specific batch size.

### 2. Inference Benchmarking

Run the following script to launch inference benchmarking on HPU:

```bash
bash dlrm_dcn_s_benchmark_hpu.sh
```

This will:

- Evaluate both RM1 and RM2 configurations
- Sweep over embedding dimensions `[4, 8, 16, ..., 512]`
- Sweep over batch sizes `[1, 2, 4, ..., 4096]`
- Store performance logs in `logs/dlrm_dcn/`

## Citation

This implementation is based on Meta’s open-source DLRM and customized for Habana Gaudi.  
Original DLRM GitHub: [https://github.com/facebookresearch/dlrm](https://github.com/facebookresearch/dlrm)

```
@article{DLRM19,
  author    = {Maxim Naumov and Dheevatsa Mudigere and Hao{-}Jun Michael Shi and Jianyu Huang and Narayanan Sundaraman and Jongsoo Park and Xiaodong Wang and Udit Gupta and Carole{-}Jean Wu and Alisson G. Azzolini and Dmytro Dzhulgakov and Andrey Mallevich and Ilia Cherniavskii and Yinghai Lu and Raghuraman Krishnamoorthi and Ansha Yu and Volodymyr Kondratenko and Stephanie Pereira and Xianjie Chen and Wenlin Chen and Vijay Rao and Bill Jia and Liang Xiong and Misha Smelyanskiy},
  title     = {Deep Learning Recommendation Model for Personalization and Recommendation Systems},
  journal   = {CoRR},
  volume    = {abs/1906.00091},
  year      = {2019},
  url       = {https://arxiv.org/abs/1906.00091},
}
```
