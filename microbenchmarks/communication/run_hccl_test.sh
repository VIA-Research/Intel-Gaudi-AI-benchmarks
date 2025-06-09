#!/bin/bash

mkdir -p result

benchmarks=("all_reduce" "all_gather" "broadcast" "reduce" "reduce_scatter")

for perf in "${benchmarks[@]}"
do
        for NRANKS in {2..8}
        do
                echo "HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks ${NRANKS} --node_id 0 --size_range 16k 1g --size_range_inc 1 --data_type bfloat16 --test ${perf} --loop 20 --ranks_per_node ${NRANKS} --csv_path result/${perf}_result_${NRANKS}.csv"
                HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py    --nranks ${NRANKS} \
                                                                        --node_id 0 \
                                                                        --size_range 1k 1g \
                                                                        --size_range_inc 1\
                                                                        --data_type bfloat16 \
                                                                        --test ${perf} \
                                                                        --loop 20 \
                                                                        --ranks_per_node ${NRANKS} \
                                                                        --csv_path result/${perf}_result_${NRANKS}.csv \
                                                                        > result/${perf}_${NRANKS}.log
        done
done

benchmarks=("all2all")
for perf in "${benchmarks[@]}"
do
        for NRANKS in 2 4 8
        do
                echo "HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks ${NRANKS} --node_id 0 --size_range 16k 1g --size_range_inc 1 --data_type bfloat16 --test ${perf} --loop 20 --ranks_per_node ${NRANKS} --csv_path result/${perf}_result_${NRANKS}.csv"
                HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py    --nranks ${NRANKS} \
                                                                        --node_id 0 \
                                                                        --size_range 1k 1g \
                                                                        --size_range_inc 1\
                                                                        --data_type bfloat16 \
                                                                        --test ${perf} \
                                                                        --loop 20 \
                                                                        --ranks_per_node ${NRANKS} \
                                                                        --csv_path result/${perf}_result_${NRANKS}.csv \
                                                                        > result/${perf}_${NRANKS}.log
        done
done