from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
import logging
from utils import adjust_batch, count_hpu_graphs, finalize_quantization, initialize_model
import os
import time
import gc

logger = logging.getLogger(__name__)

import subprocess
import threading
import time
import re

def extract_power_draw(test_string, i_l):
    power_draw_values = [float(val) for val in re.findall(r'Power Draw\s*:\s*(\d+)\s*W', test_string)]
    
    out = []
    for i in i_l:
        psu_54v = power_draw_values[2*i]
        psu_12v = power_draw_values[2*i+1]
        out.append(psu_54v + psu_12v)
        
    return out

def get_hpu_power():
    result = subprocess.run(
        # ["hl-smi", "--query-aip=index,utilization.aip,power.draw", "--format=csv,noheader,nounits"],
        ["hl-smi", "--query", "--display", "POWER"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        try:
            return result.stdout
        except ValueError:
            pass
    return ""

def monitor_power(interval, power_measurements, stop_event):
    while not stop_event.is_set():
        power_draw = get_hpu_power()
        power_measurements.append(power_draw)
        time.sleep(interval)
        

def setup_parser(parser):
    # Arguments management
    parser.add_argument("--device", "-d", type=str, choices=["hpu"], help="Device to run", default="hpu")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        help="Optional argument if you want to assess your model on a given dataset of the HF Hub.",
    )
    parser.add_argument(
        "--column_name",
        default=None,
        type=str,
        help="If `--dataset_name` was given, this will be the name of the column to use as prompts for generation.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        type=int,
        help="Size of candidate set used for re-ranking in contrastive search. top_k > 1 enables contrastive search.",
    )
    parser.add_argument(
        "--penalty_alpha",
        default=None,
        type=float,
        help="Degeneration penalty for contrastive search. penalty_alpha > 0 enables contrastive search.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--profiling_record_shapes",
        action="store_true",
        help="Record shapes when enabling profiling.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        nargs="*",
        help='Optional argument to give a prompt of your choice as input. Can be a single string (eg: --prompt "Hello world"), or a list of space-separated strings (eg: --prompt "Hello world" "How are you?")',
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument(
        "--assistant_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a draft/assistant model for assisted decoding.",
    )
    parser.add_argument(
        "--peft_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a PEFT model.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
        "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
    )
    parser.add_argument(
        "--model_revision",
        default="main",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--attn_softmax_bf16",
        action="store_true",
        help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
        "is also running in lower precision.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to store results in.",
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--dataset_max_samples",
        default=-1,
        type=int,
        help="If a negative number is passed (default = -1) perform inference on the whole dataset, else use only `dataset_max_samples` samples.",
    )
    parser.add_argument(
        "--limit_hpu_graphs",
        action="store_true",
        help="Skip HPU Graph usage for first token to save memory",
    )
    parser.add_argument(
        "--show_graphs_count",
        action="store_true",
        help="Show statistics of HPU graph compilation.",
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Whether to reuse key/value cache for decoding. It should save memory.",
    )
    parser.add_argument("--verbose_workers", action="store_true", help="Enable output from non-master workers")
    parser.add_argument(
        "--simulate_dyn_prompt",
        default=None,
        type=int,
        nargs="*",
        help="If empty, static prompt is used. If a comma separated list of integers is passed, we warmup and use those shapes for prompt length.",
    )
    parser.add_argument(
        "--reduce_recompile",
        action="store_true",
        help="Preprocess on cpu, and some other optimizations. Useful to prevent recompilations when using dynamic prompts (simulate_dyn_prompt)",
    )

    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        action="store_true",
        help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
    )
    parser.add_argument(
        "--flash_attention_fast_softmax",
        action="store_true",
        help="Whether to enable Habana Flash Attention in fast softmax mode.",
    )
    parser.add_argument(
        "--book_source",
        action="store_true",
        help="Whether to use project Guttenberg books data as input. Usefull for testing large sequence lenghts.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to use torch compiled model or not.",
    )
    parser.add_argument(
        "--ignore_eos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to disable stopping with eos token when calling `generate`. --no-ignore_eos to disable it",
    )
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature value for text generation")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top_p value for generating text via sampling")
    parser.add_argument(
        "--const_serialization_path",
        "--csp",
        type=str,
        help="Path to serialize const params. Const params will be held on disk memory instead of being allocated on host memory.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.",
    )
    parser.add_argument(
        "--parallel_strategy",
        type=str,
        choices=["tp", "none"],  # Add other strategies as needed
        default="none",
        help="Run multi card with the specified parallel strategy. Choices are 'tp' for Tensor Parallel Strategy or 'none'.",
    )
    parser.add_argument(
        "--input_embeds",
        action="store_true",
        help="Whether to enable inputs_embeds or not.",
    )
    parser.add_argument(
        "--run_partial_dataset",
        action="store_true",
        help="Run the inference with dataset for specified --n_iterations(default:5)",
    )

    ## New arguments
    parser.add_argument("--breakdown", action="store_true", default=False)
    parser.add_argument("--input-length", type=int, default=10)
    parser.add_argument("--output-length", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--merged-log-name", type=str, default="final")
    parser.add_argument("--power", action="store_true", default=False)
    ################
    
    quant_parser_group = parser.add_mutually_exclusive_group()
    quant_parser_group.add_argument(
        "--load_quantized_model_with_autogptq",
        action="store_true",
        help="Load an AutoGPTQ quantized checkpoint using AutoGPTQ.",
    )
    quant_parser_group.add_argument(
        "--disk_offload",
        action="store_true",
        help="Whether to enable device map auto. In case no space left on cpu, weights will be offloaded to disk.",
    )
    quant_parser_group.add_argument(
        "--load_quantized_model_with_inc",
        action="store_true",
        help="Load a Huggingface quantized checkpoint using INC.",
    )
    quant_parser_group.add_argument(
        "--local_quantized_inc_model_path",
        type=str,
        default=None,
        help="Path to neural-compressor quantized model, if set, the checkpoint will be loaded.",
    )

    args = parser.parse_args()

    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    if args.use_flash_attention and not args.flash_attention_fast_softmax:
        args.flash_attention_fast_softmax = True

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    if args.quant_config and args.load_quantized_model_with_autogptq:
        raise RuntimeError("Setting both quant_config and load_quantized_model_with_autogptq is unsupported. ")

    if args.quant_config == "" and args.disk_offload:
        logger.warning(
            "`--disk_offload` was tested only with fp8, it may not work with full precision. If error raises try to remove the --disk_offload flag."
        )
    return args

def run():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)
    
    warmup = args.warmup
    n_iterations = args.n_iterations
    power = args.power
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        assert False
    
    use_lazy_mode = True
    if args.torch_compile:
        use_lazy_mode = False
        
    if True:
        args.prompt = "Here is my prompt: Vertically Integrated Architecture (VIA) research group is affiliated with the School of Electrical Engineering, Department of Semiconductor System Engineering, Graduate School of Artificial Intelligence (AI), and Graduate School of AI Semiconductor at KAIST, South Korea. We conduct research in the domain of computer architecture with a vertically integrated approach. By co-optimizing VLSI technology, computer system architecture, and application & algorithms, our mission is to build a high-performance computing platform for future \"intelligent\" systems that are programmable, robust, reliable, secure, and energy-efficient. (Note) For students interested in undergraduate research internships or those who are applying to our research group for graduate studies, please send me an email with your latest CV and transcript. I'm a Tenured Associate Professor at the School of Electrical Engineering, jointly affiliated with Department of Semiconductor System Engineering,  Graduate School of Artificial Intelligence (AI), and Graduate School of  AI Semiconductor at KAIST. I am was a Senior Research Scientist working at the architecture research group at NVIDIA. I had an opportunity to work on a number of exciting projects at NVIDIA that span several areas in computing, which include ASIC designs, computer system architecture, runtime systems, and application & workload characterization with an emphasis on deep neural networks (DNNs). Initially at NVIDIA, I worked on developing microarchitectural support for high-performance GPU cache replacement policies. More recently, I have been working in the domain of deep learning, trying to come up with neat architectural enhancements to the GPU hardware/software stack so that NVIDIA maintains its leadership in the areas of machine learning. For instance, I led the research and development of the virtualized DNN runtime system, a high-performance GPU memory virtualization solution for DNN training. I was also the technical lead on the architecture design, implementation, and evaluation of the sparse CNN accelerator, an ASIC developed by NVIDIA Research aiming towards achieving high energy-efficiency for DNN inference. In the past, I earned my Ph.D. degree from the University of Texas at Austin in 2014, under the guidance of professor Mattan Erez. I received my M.S. and B.E. degree from KAIST (Korea Advanced Institute of Science and Technology) and Sogang University, in 2009 and 2007, respectively. On the 5th, KAIST announced that Minsoo Rhu, Professor in the Department of Electrical Engineering, has been appointed as Program Co-Chair for the IEEE/ACM International Symposium on Microarchitecture (MICRO) scheduled to be held next year. This marks the first time in MICRO’s 57-year history that a faculty member from an Asian university has been selected as Program Chair. Celebrating its 57th edition this year, MICRO is the oldest and most prestigious international conference in the field of computer architecture. Alongside ISCA and HPCA, it is regarded as one of the top three international conferences in computer architecture. Scholars and industry professionals from around the world participate in MICRO, with fewer than 20% of submitted papers being selected for final presentation. Professor Rhu was appointed Program Chair of the 58th MICRO conference, set to be held next year, in recognition of his contributions to the field of computer architecture. He will serve as Program Co-Chair alongside Professor Radu Teodorescu of Ohio State University, overseeing the selection of around 300 expert members of the Program Committee and supervising the review of over 500 submitted papers. Professor Rhu is recognized as a next-generation leader in the fields of intelligent semiconductors and computer systems for artificial intelligence (AI). His expertise is reflected in his induction into the Hall of Fame of major conferences, including HPCA in 2021, MICRO in 2022, and ISCA this year. Professor Rhu completed his undergraduate studies in electronic engineering at Sogang University, obtained his master’s degree in electrical engineering from KAIST, and earned his Ph.D. in computer science from the University of Texas at Austin. From 2014 to 2017, he worked at NVIDIA Research, and since 2018, he has been a professor at KAIST. He also served as a visiting researcher at Meta AI from 2022 to last year. His research has been widely recognized by academia, receiving the Best Paper Award at HPCA this year, the Google Faculty Research Award last year, and the Facebook Faculty Research Award in 2020. Last year, he was also inducted as a member of Y-KAST, an elite group of young scientists under 43 recognized for their outstanding contributions to science by the Korean Academy of Science and Technology."
    
    model, assistant_model, tokenizer, generation_config = initialize_model(args, logger)
    generation_config.top_p = None
    generation_config.temperature = None
    assert model.dtype == torch.bfloat16
    assert dtype == torch.bfloat16
    
    import habana_frameworks.torch.hpu as torch_hpu
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]    
    input_output_lengths = [(100, 25), (100, 50), (100, 100), (100, 200), (100, 400)]
    
    latency_list = []
    throughput_list = []
    input_length_list = []
    output_length_list = []
    batch_list = [] 
    power_list = []
    splitted_model_name = args.model_name_or_path.split("/")
    if splitted_model_name[-1] == "":
        model_name = splitted_model_name[-2]
    else:
        model_name = splitted_model_name[-1]
    
    
    with torch.no_grad():
        for input_length, output_length in input_output_lengths:
            # Adjust the input length
            tmp_input_tokens = tokenizer.encode(args.prompt)
            assert len(tmp_input_tokens) >= input_length
            final_prompt = tokenizer.decode(tmp_input_tokens[:input_length], skip_special_tokens=True)
        
            args.max_new_tokens = output_length
            generation_config.max_new_tokens = output_length
            for batch_size in batch_sizes:
                
                log_path = "./logs/%s_I_%d_O_%d_B_%d.txt" %(model_name, input_length, output_length, batch_size)
                log_path = log_path[:-4] + "_%dHPUs.txt" %args.world_size
                    
                print("Experiment for %s" %log_path)
                
                f = open(log_path, 'w')
                
                # Make batched prompt
                batched_prompt = []
                for _ in range(batch_size):
                    batched_prompt.append(final_prompt)
                    
                breakdown_times = None
                iteration_times = []
                
                if power:
                    power_measurements = []
                    stop_event = threading.Event()
                    power_thread = threading.Thread(target=monitor_power, args=(0.001, power_measurements, stop_event))
                    power_thread.start()
                try:
                    start_time = time.perf_counter()
                    while time.perf_counter() - start_time < 60:
                        for iter in range(warmup + n_iterations):
                            torch_hpu.synchronize()
                            start = time.perf_counter()
                                
                            input_tokens = tokenizer.batch_encode_plus(batched_prompt, return_tensors="pt", padding=True)
                            assert len(input_tokens["input_ids"]) == batch_size
                            assert len(input_tokens["input_ids"][0]) == input_length
                            
                            for t in input_tokens:
                                if torch.is_tensor(input_tokens[t]):
                                    input_tokens[t] = input_tokens[t].to(args.device)
                            
                            if args.breakdown:
                                decode_step_times = []
                            else:
                                decode_step_times = None
                                
                            
                            output_tokens = model.generate(
                                        **input_tokens,
                                        generation_config=generation_config,
                                        assistant_model=assistant_model,
                                        do_sample=False,
                                        lazy_mode=use_lazy_mode,
                                        hpu_graphs=args.use_hpu_graphs,
                                        profiling_steps=args.profiling_steps,
                                        profiling_warmup_steps=args.profiling_warmup_steps,
                                        ignore_eos=args.ignore_eos,
                                        iteration_times=decode_step_times,
                                        breakdown_times=breakdown_times,
                                        profiling_record_shapes=args.profiling_record_shapes,
                                    ).cpu()
                            
                            output_sentences = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
                            
                            torch_hpu.synchronize()
                            end = time.perf_counter()
                            
                            iteration_times.append(end - start)
                            
                            
                except Exception as error:
                    f.write("Error name: %s\n" %type(error).__name__)
                    f.write("\n")
                    f.write("Error message:\n")
                    f.write(str(error))
                    f.close()
                    latency_list.append(0) # msec
                    throughput_list.append(0)
                    input_length_list.append(input_length)
                    output_length_list.append(output_length)
                    batch_list.append(batch_size)
                    if power:
                        power_list.append([0])
                        stop_event.set()
                        power_thread.join()
                    
                    if args.use_hpu_graphs:
                        model.clear_cache()
                    gc.collect()
                    continue
                
                if power:                
                    stop_event.set()
                    power_thread.join()
                            
                duration = sum(iteration_times[-n_iterations:])/n_iterations
                total_new_tokens_generated = batch_size * output_length
                throughput = total_new_tokens_generated / duration
                
                f.write("Results:\n")
                f.write(output_sentences[0])
                f.write("\n")
                f.write("Batch size: %d / Input length: %d / output length: %d\n" %(len(batched_prompt), len(input_tokens[0]), len(output_tokens[0])))
                f.write("Iteration itmes: ")
                for i in range(len(iteration_times)):
                    f.write("%f, " %iteration_times[i])

                # print(iteration_times)
                if power:
                    if args.world_size == 0:
                        i_l = [6]
                    elif args.world_size == 2:
                        i_l = [6, 7]
                    elif args.world_size == 4:
                        i_l = [4, 5, 6, 7]
                    elif args.world_size == 8:
                        i_l = [0, 1, 2, 3, 4, 5, 6, 7]
                    else:
                        assert False
                        
                    tensor_data = []
                    for measurement in power_measurements:
                        # rows = measurement.strip().split("\n")
                        # parsed_rows = [list(map(float, row.split(","))) for row in rows]
                        # tensor_data.append(parsed_rows)
                        
                        tensor_data.append(extract_power_draw(measurement, i_l))

                    result_tensor = torch.tensor(tensor_data)
                    # sum_util = result_tensor[:, :, 1].sum(dim=0)
                    final_power = []
                    
                    # for i in range(sum_util.shape[0]):
                        # if sum_util[i] != 0:
                        #     final_power.append(result_tensor[:, i, 2].mean().item())
                    
                    for i in range(len(i_l)):
                        final_power.append(result_tensor[:, i].mean().item())
                        
                    if args.world_size == 0:
                        assert len(final_power) == 1
                    else:
                        assert len(final_power) == args.world_size
                        
                    # f.write("Powers: ")
                    # for i in range(sum_util.shape[0]):
                    #     if sum_util[i] != 0:
                    #         for j in range(result_tensor.shape[0]):
                    #             f.write("%f, " %result_tensor[j][i][2])
                    #         f.write("\n\n")
                    
                    f.write("Powers: ")
                    for i in range(len(i_l)):
                        for j in range(result_tensor.shape[0]):
                            f.write("%f, " %result_tensor[j][i])
                        f.write("\n\n")
                    
                f.write("\n\n")
                if not power:
                    f.write("Latency (msec), Throughput (tokens/sec)\n")
                    f.write("%f, %f" %(duration * 1000, throughput))
                else:
                    label = "Latency (msec), Throughput (tokens/sec), "
                    value =  "%f, %f, " %(duration * 1000, throughput)
                    for i in range(len(final_power)):
                        label = label + "Power %d" %i
                        value = value + "%f" %final_power[i]
                        if i == len(final_power) - 1:
                            label = label + "\n"
                        else:
                            label = label + ", "
                            value = value + ", "
                    f.write(label)
                    f.write(value)
                f.close()
                
                latency_list.append(duration * 1000) # msec
                throughput_list.append(throughput)
                input_length_list.append(input_length)
                output_length_list.append(output_length)
                batch_list.append(batch_size)
                if power:
                    power_list.append(final_power)
                for i in output_tokens:
                    assert len(i) == output_length + input_length

                if args.use_hpu_graphs:
                    model.clear_cache()
                gc.collect()
    
    assert len(latency_list) == len(input_output_lengths) * len(batch_sizes)
    assert len(throughput_list) == len(input_output_lengths) * len(batch_sizes)
    assert len(input_length_list) == len(input_output_lengths) * len(batch_sizes)
    assert len(output_length_list) == len(input_output_lengths) * len(batch_sizes)
    assert len(batch_list) == len(input_output_lengths) * len(batch_sizes)
    
    final_path = "./logs/final_%s_%s_%dHPUs.txt" %(args.merged_log_name, model_name, args.world_size)
        
    f = open(final_path, 'w')
    if not power:
        f.write("Input_len, output_len, batch_size, latency (msec), throughput (token/sec)\n")
        for i in range(len(batch_list)):
            f.write("%d, %d, %d, %f, %f\n" %(input_length_list[i], output_length_list[i], batch_list[i], latency_list[i], throughput_list[i]))
    else:
        label = "Input_len, output_len, batch_size, latency (msec), throughput (token/sec), "
        if args.world_size == 0:
            n_hpu = 1
        else:
            n_hpu = args.world_size
            
            
        for i in range(n_hpu):
            label = label + "Power %d" %i
            if i == n_hpu - 1:
                label = label + "\n"
            else:
                label = label + ", "
        f.write(label)
                
        for i in range(len(batch_list)):
            value =  "%d, %d, %d, %f, %f, " %(input_length_list[i], output_length_list[i], batch_list[i], latency_list[i], throughput_list[i])
            for j in range(len(power_list[i])):
                label = label + "Power %d" %j
                value = value + "%f" %power_list[i][j]
                if j == len(power_list[i]) - 1:
                    value = value + "\n"
                else:
                    value = value + ", "
            f.write(value)
        
    f.close()
        
if __name__ == "__main__":
    run()
