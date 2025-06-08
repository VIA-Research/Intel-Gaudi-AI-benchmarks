#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
dtype=$5     # dtype: fp16, fp32, bf16

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$dtype" ]]; then
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

echo "Generated values for M, K, N: ${M_values[@]}"

# Run the loop for M, K, N values
for value in "${M_values[@]}"
do
	output_file="./trace_files/${value}_${value}_${value}_${dtype}.csv"
	if [ -f "$output_file" ]; then
        echo "File $output_file already exists. Skipping..."
        continue
    fi

    echo $value, $value, $value
    python gemm.py --m=$value --k=$value --n=$value --dtype=$dtype
    mv *_analyzed_nodes.csv ${value}_${value}_${value}_${dtype}.csv
	mv ${value}_${value}_${value}_${dtype}.csv ./trace_files
    rm default_*
done
