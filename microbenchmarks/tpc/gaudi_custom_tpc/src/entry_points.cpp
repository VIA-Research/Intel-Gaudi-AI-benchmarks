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

#include "add_bf16_gaudi2.hpp"
#include "add_bf16_unroll1_gaudi2.hpp"
#include "add_bf16_unroll2_gaudi2.hpp"
#include "add_bf16_unroll3_gaudi2.hpp"
#include "add_Nop_bf16_gaudi2.hpp"
#include "scale_bf16_gaudi2.hpp"
#include "scale_bf16_unroll1_gaudi2.hpp"
#include "scale_bf16_unroll2_gaudi2.hpp"
#include "scale_bf16_unroll3_gaudi2.hpp"
#include "scale_Nop_bf16_gaudi2.hpp"
#include "triad_bf16_gaudi2.hpp"
#include "triad_bf16_unroll1_gaudi2.hpp"
#include "triad_bf16_unroll2_gaudi2.hpp"
#include "triad_bf16_unroll3_gaudi2.hpp"
#include "triad_Nop_bf16_gaudi2.hpp"
#include "gather_bf16_gaudi2.hpp"
#include "scatter_bf16_gaudi2.hpp"
#include "entry_points.hpp"
#include <stdio.h>
#include <cstring>

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
           AddBF16Gaudi2 addbf16g2Instance;
           addbf16g2Instance.GetKernelName(guids[GAUDI2_KERNEL_ADD_BF16].name);
           AddBF16Unroll1Gaudi2 addbf16unroll1g2Instance;
           addbf16unroll1g2Instance.GetKernelName(guids[GAUDI2_KERNEL_ADD_BF16_UNROLL1].name);
           AddBF16Unroll2Gaudi2 addbf16unroll2g2Instance;
           addbf16unroll2g2Instance.GetKernelName(guids[GAUDI2_KERNEL_ADD_BF16_UNROLL2].name);
           AddBF16Unroll3Gaudi2 addbf16unroll3g2Instance;
           addbf16unroll3g2Instance.GetKernelName(guids[GAUDI2_KERNEL_ADD_BF16_UNROLL3].name);
           AddNOpBF16Gaudi2 addNopbf16g2Instance;
           addNopbf16g2Instance.GetKernelName(guids[GAUDI2_KERNEL_ADD_N_OP_BF16].name);
           
           ScaleBF16Gaudi2 scalebf16g2Instance;
           scalebf16g2Instance.GetKernelName(guids[GAUDI2_KERNEL_SCALE_BF16].name);
           ScaleBF16Unroll1Gaudi2 scalebf16unroll1g2Instance;
           scalebf16unroll1g2Instance.GetKernelName(guids[GAUDI2_KERNEL_SCALE_BF16_UNROLL1].name);
           ScaleBF16Unroll2Gaudi2 scalebf16unroll2g2Instance;
           scalebf16unroll2g2Instance.GetKernelName(guids[GAUDI2_KERNEL_SCALE_BF16_UNROLL2].name);
           ScaleBF16Unroll3Gaudi2 scalebf16unroll3g2Instance;
           scalebf16unroll3g2Instance.GetKernelName(guids[GAUDI2_KERNEL_SCALE_BF16_UNROLL3].name);
           ScaleNOpBF16Gaudi2 scaleNopbf16g2Instance;
           scaleNopbf16g2Instance.GetKernelName(guids[GAUDI2_KERNEL_SCALE_N_OP_BF16].name);

           TriadBF16Gaudi2 triadbf16g2Instance;
           triadbf16g2Instance.GetKernelName(guids[GAUDI2_KERNEL_TRIAD_BF16].name);
           TriadBF16Unroll1Gaudi2 triadbf16unroll1g2Instance;
           triadbf16unroll1g2Instance.GetKernelName(guids[GAUDI2_KERNEL_TRIAD_BF16_UNROLL1].name);
           TriadBF16Unroll2Gaudi2 triadbf16unroll2g2Instance;
           triadbf16unroll2g2Instance.GetKernelName(guids[GAUDI2_KERNEL_TRIAD_BF16_UNROLL2].name);
           TriadBF16Unroll3Gaudi2 triadbf16unroll3g2Instance;
           triadbf16unroll3g2Instance.GetKernelName(guids[GAUDI2_KERNEL_TRIAD_BF16_UNROLL3].name);
           TriadNOpBF16Gaudi2 triadNopbf16g2Instance;
           triadNopbf16g2Instance.GetKernelName(guids[GAUDI2_KERNEL_TRIAD_N_OP_BF16].name);

           GatherBF16Gaudi2 gatherbf16g2Instance;
           gatherbf16g2Instance.GetKernelName(guids[GAUDI2_KERNEL_GATHER_BF16].name);

           ScatterBF16Gaudi2 scatterbf16g2Instance;
           scatterbf16g2Instance.GetKernelName(guids[GAUDI2_KERNEL_SCATTER_BF16].name);
        }

        if (kernelCount != nullptr)
        {
            // currently the library support 8 kernel.
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

    /////////////////////////////////////////////////////////////////////////
    AddBF16Gaudi2 addbf16g2Instance;
    addbf16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return addbf16g2Instance.GetGcDefinitions(params, instance);
    }
    AddBF16Unroll1Gaudi2 addbf16unroll1g2Instance;
    addbf16unroll1g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return addbf16unroll1g2Instance.GetGcDefinitions(params, instance);
    }
    AddBF16Unroll2Gaudi2 addbf16unroll2g2Instance;
    addbf16unroll2g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return addbf16unroll2g2Instance.GetGcDefinitions(params, instance);
    }
    AddBF16Unroll3Gaudi2 addbf16unroll3g2Instance;
    addbf16unroll3g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return addbf16unroll3g2Instance.GetGcDefinitions(params, instance);
    }
    AddNOpBF16Gaudi2 addNopbf16g2Instance;
    addNopbf16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return addNopbf16g2Instance.GetGcDefinitions(params, instance);
    }

    /////////////////////////////////////////////////////////////////////////
    ScaleBF16Gaudi2 scalebf16g2Instance;
    scalebf16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return scalebf16g2Instance.GetGcDefinitions(params, instance);
    }
    ScaleBF16Unroll1Gaudi2 scalebf16unroll1g2Instance;
    scalebf16unroll1g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return scalebf16unroll1g2Instance.GetGcDefinitions(params, instance);
    }
    ScaleBF16Unroll2Gaudi2 scalebf16unroll2g2Instance;
    scalebf16unroll2g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return scalebf16unroll2g2Instance.GetGcDefinitions(params, instance);
    }
    ScaleBF16Unroll3Gaudi2 scalebf16unroll3g2Instance;
    scalebf16unroll3g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return scalebf16unroll3g2Instance.GetGcDefinitions(params, instance);
    }
    ScaleNOpBF16Gaudi2 scaleNopbf16g2Instance;
    scaleNopbf16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return scaleNopbf16g2Instance.GetGcDefinitions(params, instance);
    }

    /////////////////////////////////////////////////////////////////////////
    TriadBF16Gaudi2 triadbf16g2Instance;
    triadbf16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return triadbf16g2Instance.GetGcDefinitions(params, instance);
    }
    TriadBF16Unroll1Gaudi2 triadbf16unroll1g2Instance;
    triadbf16unroll1g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return triadbf16unroll1g2Instance.GetGcDefinitions(params, instance);
    }
    TriadBF16Unroll2Gaudi2 triadbf16unroll2g2Instance;
    triadbf16unroll2g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return triadbf16unroll2g2Instance.GetGcDefinitions(params, instance);
    }
    TriadBF16Unroll3Gaudi2 triadbf16unroll3g2Instance;
    triadbf16unroll3g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return triadbf16unroll3g2Instance.GetGcDefinitions(params, instance);
    }
    TriadNOpBF16Gaudi2 triadNopbf16g2Instance;
    triadNopbf16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return triadNopbf16g2Instance.GetGcDefinitions(params, instance);
    }

    GatherBF16Gaudi2 gatherbf16g2Instance;
    gatherbf16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return gatherbf16g2Instance.GetGcDefinitions(params, instance);
    }

    ScatterBF16Gaudi2 scatterbf16g2Instance;
    scatterbf16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->guid.name, kernelName) == 0)
    {
        return scatterbf16g2Instance.GetGcDefinitions(params, instance);
    }

    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId deviceId,  tpc_lib_api::ShapeInferenceParams* inputParams,  tpc_lib_api::ShapeInferenceOutput* outputData)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

} // extern "C"
