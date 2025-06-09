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

#include "embedding_bag_sum_f32_gaudi2.hpp"
#include "table_batched_embedding_bag_sum_f32_gaudi2.hpp"

#include "entry_points.hpp"
#include <cstring>
#include <stdio.h>
extern "C"
{

tpc_lib_api::GlueCodeReturn GetKernelGuids( _IN_    tpc_lib_api::DeviceId        deviceId,
                                            _INOUT_ uint32_t*       kernelCount,
                                            _OUT_   tpc_lib_api::GuidInfo*       guids)
{

    if (deviceId == tpc_lib_api::DEVICE_ID_GAUDI2)
    {
        if (guids != nullptr )
        {
           EmbeddingBagSumF32Gaudi2 embeddingbagsumf32g2Instance;
           embeddingbagsumf32g2Instance.GetKernelName(guids[GAUDI2_KERNEL_EMBEDDING_BAG_SUM_F32].name);
           TableBatchedEmbeddingBagSumF32Gaudi2 tablebatchedembeddingbagsumf32g2Instance;
           tablebatchedembeddingbagsumf32g2Instance.GetKernelName(guids[GAUDI2_KERNEL_TABLE_BATCHED_EMBEDDING_BAG_SUM_F32].name);
        }

        if (kernelCount != nullptr)
        {
            *kernelCount = GAUDI2_KERNEL_MAX_EXAMPLE_KERNEL;
        }
    }
    else
    {
        if (kernelCount != nullptr)
        {
            // currently the library support 0 kernels.
            *kernelCount = 0;
        }
    }
    return tpc_lib_api::GLUE_SUCCESS;
}


tpc_lib_api::GlueCodeReturn
InstantiateTpcKernel(_IN_  tpc_lib_api::HabanaKernelParams* params,
             _OUT_ tpc_lib_api::HabanaKernelInstantiation* instance)
{
    char kernelName [tpc_lib_api::MAX_NODE_NAME];

    /////// --- Gaudi2 
    ///////////////////////////////

    EmbeddingBagSumF32Gaudi2 embeddingbagsumf32g2Instance;
    embeddingbagsumf32g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return embeddingbagsumf32g2Instance.GetGcDefinitions(params,instance);
    }

    TableBatchedEmbeddingBagSumF32Gaudi2 tablebatchedembeddingbagsumf32g2Instance;
    tablebatchedembeddingbagsumf32g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return tablebatchedembeddingbagsumf32g2Instance.GetGcDefinitions(params,instance);
    }

    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId deviceId,  tpc_lib_api::ShapeInferenceParams* inputParams,  tpc_lib_api::ShapeInferenceOutput* outputData)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

} // extern "C"
