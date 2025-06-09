/**********************************************************************
Copyright (c) 2024 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <vector>
#include <cstring>
#include <iostream>
#include "table_batched_embedding_bag_sum_f32_gaudi2.hpp"


extern unsigned char _binary___table_batched_embedding_bag_sum_f32_gaudi2_o_start;
extern unsigned char _binary___table_batched_embedding_bag_sum_f32_gaudi2_o_end;

 tpc_lib_api::GlueCodeReturn TableBatchedEmbeddingBagSumF32Gaudi2::GetKernelName(
             char kernelName [tpc_lib_api::MAX_NODE_NAME])
 {
     strcpy(kernelName,"custom_table_batched_embedding_bag_sum_f32_gaudi2");
     return tpc_lib_api::GLUE_SUCCESS;
 }


tpc_lib_api::GlueCodeReturn TableBatchedEmbeddingBagSumF32Gaudi2::GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* in_defs,
            tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
    tpc_lib_api::GlueCodeReturn retVal;
    // const int c_unrollCount = 4;
    TableBatchedEmbeddingBagSumF32Param* def = static_cast<TableBatchedEmbeddingBagSumF32Param*>(in_defs->nodeParams.nodeParams);

    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 4)
    {
        in_defs->inputTensorNr  = 4;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->inputTensors[1].geometry.dataType != tpc_lib_api::DATA_I32 ||
        in_defs->inputTensors[2].geometry.dataType != tpc_lib_api::DATA_I32 ||
        in_defs->inputTensors[3].geometry.dataType != tpc_lib_api::DATA_I32 ||
        in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        in_defs->inputTensors[1].geometry.dataType = tpc_lib_api::DATA_I32;
        in_defs->inputTensors[2].geometry.dataType = tpc_lib_api::DATA_I32;
        in_defs->inputTensors[3].geometry.dataType = tpc_lib_api::DATA_I32;
        in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    out_defs->indexSpaceRank = 1;
    out_defs->indexSpaceGeometry[0] = def->n_tpc;

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/

    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr = sizeof(*def) / sizeof(int);
    memcpy(&(out_defs->kernel.scalarParams[0]), def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___table_batched_embedding_bag_sum_f32_gaudi2_o_end - &_binary___table_batched_embedding_bag_sum_f32_gaudi2_o_start);
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;

    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy (out_defs->kernel.kernelElf,
                &_binary___table_batched_embedding_bag_sum_f32_gaudi2_o_start,
                IsaSize);
    }
    else
    {
       retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
       return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

