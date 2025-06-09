# Intel Gaudi LLM Benchmark

This repository provides scripts and helper tools to benchmark LLM text-generation performance using Hugging Face’s Optimum-Habana on Intel Gaudi NPUs. It supports multi-device runs (1, 2, 4, 8 Gaudi devices).

---

### Prerequisites

1. **Clone Optimum-Habana**  
   ```bash
   git clone https://github.com/huggingface/optimum-habana.git
   ```
2. **Move files in current repository to optimum-hanaban repository**
    ```bash
   mv run_various_configs.py run_entire_experiments.sh optimum-habana/examples/text-generation
   ```
3. **Follow the requirements specified in the Optimum-Habana repository**
4. **Download the Hugging Face LLM models (e.g., meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.1-70B-Instruct)**
---

### Repository Structure
```bash
.
├── run_various_configs.py
├── run_entire_experiments.sh
└── logs/
```

- **run_various_configs.py**  
  - Launches text-generation experiments for a given model/config on 1, 2, 4, and 8 Gaudi devices.
  - - Usage:  
    ```bash
    # Run LLM with single device
    python run_various_configs.py --model_name_or_path {path_to_model} --use_kv_cache --ignore_eos --dtype bf16 --bf16 --use_hpu_graphs --use_flash_attention --flash_attention_recompute --flash_attention_causal_mask --reuse_cache --attn_softmax_bf16 --merged-log-name {log_name}
    
    # Run LLM with multiple devices
    python ../gaudi_spawn.py --use_deepspeed --world_size {n_devices} run_various_configs.py --model_name_or_path {path_to_model} --use_kv_cache --ignore_eos --dtype bf16 --bf16 --use_hpu_graphs --use_flash_attention --flash_attention_recompute --flash_attention_causal_mask --reuse_cache --attn_softmax_bf16 --merged-log-name {log_name}
    ```  
- **run_entire_experiments.sh**  
  - Shell wrapper that calls `run_various_configs.py` sequentially for each device count.  
