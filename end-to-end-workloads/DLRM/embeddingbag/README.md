# Gaudi-v2 kernel for EmbeddingBag operation

This repository provides the TPC kernel for custom embeddingbag operation using Gaudi-v2. This example is implemented based on custom TPC kernel examples (https://github.com/HabanaAI/Habana_Custom_Kernel/tree/main).

## Install Habanatools For Ubuntu
To retrieve the package please visit [Habana Vault](https://vault.habana.ai/artifactory/debian/jammy/pool/main/h/habanatools/habanatools_1.18.0-524_amd64.deb), click Artifact, find habanatools and download the latest release package for Ubuntu 22.04. You can find different packages for different OS you used. 
```  
sudo dpkg -i ./habanatools_1.18.0-524_amd64.deb
```

## Build
In the terminal, make sure you are in the project root directory, then create a directory called build
```  
mkdir build
cd build
```  
then run the following commands
```  
cmake ..
make
```  
After build, you can find libcustom_tpc_perf_lib.so in build/src directory, which is your custom kernel library. For graph compiler to capture your custom kernel, run the following command.
```  
export GC_KERNEL_PATH=/path_to_your_custom_tpc_lib/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
```  

## PyTorch extension
To use the custom TPC kernel in PyTorch environment, you have to extend it to PyTorch custom operation.
```  
cd pytorch_extension/pytorch_custom_op
python setup.py build
```  
then set up the path
```  
export PYTHONPATH=/path_to_the_project_root_directory/pytorch_extension/pytorch_custom_op:$PYTHONPATH
``` 

## Test and example
To test the custom embeddingbag kernel in PyTorch environment, you have to finish the above steps.
```  
cd pytorch_extension/pytorch_custom_op
python hpu_custom_op_embedding_bag_sum.py
python hpu_custom_op_table_batched_embedding_bag_sum.py
