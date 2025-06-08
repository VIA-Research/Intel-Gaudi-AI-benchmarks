#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
dtype=$5     # dtype: fp16, fp32, bf16
fixed_size=$6

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$dtype" || -z "$fixed_size" ]]; then
    echo "Usage: $0 <type: linear|power> <start> <end> <stride> <dtype>"
    exit 1
fi

# Generate values based on type (linear or power)
if [ "$type" == "linear" ]; then
    M_values=()
    for ((i=start; i<=end; i+=stride)); do
        M_values+=($i)
    done
elif [ "$type" == "power" ]; then
    M_values=()
    value=$start
    while [ $value -le $end ]; do
        M_values+=($value)
        value=$((value * stride))
    done
else
    echo "Invalid type. Please choose 'linear' or 'power'."
    exit 1
fi

# Use the same values for K and N
N_values=("${M_values[@]}")

echo "Generated values for M, N: ${M_values[@]} with fixed K=${fixed_size}"

# Run the loop for M, K, N values
for M in "${M_values[@]}"
do
    for N in "${N_values[@]}"
    do
        output_file="./trace_files/${M}_${fixed_size}_${N}_${dtype}.csv"
        if [ -f "$output_file" ]; then
            echo "File $output_file already exists. Skipping..."
            continue
        fi
        
        echo $M, $fixed_size, $N
        python gemm.py --m=$M --k=$fixed_size --n=$N --dtype=$dtype
        mv *_analyzed_nodes.csv ${M}_${fixed_size}_${N}_${dtype}.csv
        mv ${M}_${fixed_size}_${N}_${dtype}.csv ./trace_files
        rm default_*
    done
done

