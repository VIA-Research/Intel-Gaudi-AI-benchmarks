# Intel Gaudi-2 Custom TPC Kernel Implementation

This repository provides a custom TPC (Tensor Processing Core) kernel implementation for Intel Gaudi-2's gather/scatter and stream benchmarks.
The implementation is based on Habana's official [custom TPC kernel examples](https://github.com/HabanaAI/Habana_Custom_Kernel/tree/main).

## Prerequisites

- Intel Gaudi-2 system
- [Habana® SynapseAI™ software stack](https://github.com/HabanaAI/Habana_Custom_Kernel)
- Python 3.10+
- CMake

## Install Habanatools
Follow the installation guide provided in the [Habana Custom Kernel GitHub repository](https://github.com/HabanaAI/Habana_Custom_Kernel/tree/main) to set up the required environment.

## Building the Custom TPC Kernel
In the terminal, make sure you are in the project root directory, then create a directory called build
```  
mkdir build
cd build
cmake ..
make
```  
After build, you can find libcustom_tpc_perf_lib.so in build/src directory, which is your custom kernel library. For Habana graph compiler to capture your custom kernel, run the following command.
```  
export GC_KERNEL_PATH=/path/to/your/build/src/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
```  

## PyTorch Extension
To use the custom TPC kernel in PyTorch environment, you have to extend it to PyTorch custom operation.
```  
cd pytorch_extension/pytorch_custom_op
python setup.py build
```  
then set up the path
```  
export PYTHONPATH=/path/to/project/pytorch_extension/pytorch_custom_op:$PYTHONPATH
``` 