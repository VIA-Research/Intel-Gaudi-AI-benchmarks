###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

from setuptools import setup
from torch.utils import cpp_extension
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
import os
import pybind11

torch_include_dir = get_include_dir()
torch_lib_dir = get_lib_dir()
habana_modules_directory = "/usr/include/habanalabs"
pybind_include_path = pybind11.get_include()

ops=['add', 'triad', 'scale']
dtypes=['bf16']
unrolls=['1', '2', '3']
for op in ops:
      for dtype in dtypes:
            name='src/hpu_custom_'+op+'_'+dtype
            setup(name=name,
                  ext_modules=[cpp_extension.CppExtension(name, [name+'.cpp'],
                        language='c++', extra_compile_args=["-std=c++17"],
                        libraries=['habana_pytorch_plugin'],
                        library_dirs=[torch_lib_dir])],
                  include_dirs=[torch_include_dir,
                              habana_modules_directory,
                              pybind_include_path,
                              ],
                  cmdclass={'build_ext': cpp_extension.BuildExtension})
            for unroll in unrolls:
                  name='src/hpu_custom_'+op+'_'+dtype+'_unroll'+unroll
                  setup(name=name,
                        ext_modules=[cpp_extension.CppExtension(name, [name+'.cpp'],
                              language='c++', extra_compile_args=["-std=c++17"],
                              libraries=['habana_pytorch_plugin'],
                              library_dirs=[torch_lib_dir])],
                        include_dirs=[torch_include_dir,
                                    habana_modules_directory,
                                    pybind_include_path,
                                    ],
                        cmdclass={'build_ext': cpp_extension.BuildExtension})

ops=['add_Nop', 'triad_Nop', 'scale_Nop', 'gather', 'scatter']
dtypes=['bf16']
for op in ops:
      for dtype in dtypes:
            name='src/hpu_custom_'+op+'_'+dtype
            setup(name=name,
                  ext_modules=[cpp_extension.CppExtension(name, [name+'.cpp'],
                        language='c++', extra_compile_args=["-std=c++17"],
                        libraries=['habana_pytorch_plugin'],
                        library_dirs=[torch_lib_dir])],
                  include_dirs=[torch_include_dir,
                              habana_modules_directory,
                              pybind_include_path,
                              ],
                  cmdclass={'build_ext': cpp_extension.BuildExtension})