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

void main(tensor input, tensor indices, int output_size, int n_tpc, tensor output)
{
    const int depth   = 0;
    const int width   = 1;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd   = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords0 = {0};
    int5 ifmCoords1 = {0};
    int5 ifmCoords2 = {0};
    int5 ifmCoords3 = {0};

    int5 idxCoords0 = {0};
    int5 idxCoords1 = {0};
    int5 idxCoords2 = {0};
    int5 idxCoords3 = {0};

    int5 dataCoords0 = {0};
    int5 dataCoords1 = {0};
    int5 dataCoords2 = {0};
    int5 dataCoords3 = {0};

    const int inputDepthStep   = 128;
    const int indiceDepthStep  = 64;

    // Get the number of indices
    const int indxSize = get_dim_size(indices, 0);
    const int inputDepthSize = get_dim_size(input, 0);
    
    // Use ceiling division to ensure all elements are divided across PEs
    int unit = (indxSize + n_tpc - 1) / n_tpc;

    const int IndiceDepthStart  = indexSpaceStart[depth] * unit;
    const int IndiceDepthEnd    = s_i32_min(indexSpaceEnd[depth] * unit, indxSize);  // Ensure not to exceed indices size
    
    const int inputDepthStart  = 0;
    const int inputDepthEnd  = inputDepthSize;
    
    int actualIndiceDepthStep = s_i32_min(indiceDepthStep, IndiceDepthEnd - IndiceDepthStart);

    bfloat128 x0, x1, x2, x3;

    #pragma loop_taken
    for (int d = IndiceDepthStart; d < IndiceDepthEnd; d += indiceDepthStep)
    {
        int vecEnd = s_i32_min(actualIndiceDepthStep, indxSize-d);
        int vecTail = vecEnd & 3;
        int vecEndUnrolled = vecEnd - vecTail;

        idxCoords0[depth] = d;
        idxCoords1[depth] = d + 1;
        idxCoords2[depth] = d + 2;
        idxCoords3[depth] = d + 3;

        __global__ int* idxAddr0 = ( __global__ int*)gen_addr(idxCoords0, indices); idxCoords0[depth] += 4;
        __global__ int* idxAddr1 = ( __global__ int*)gen_addr(idxCoords1, indices); idxCoords1[depth] += 4;
        __global__ int* idxAddr2 = ( __global__ int*)gen_addr(idxCoords2, indices); idxCoords2[depth] += 4;
        __global__ int* idxAddr3 = ( __global__ int*)gen_addr(idxCoords3, indices); idxCoords3[depth] += 4;

        for (int vec_d = 0; vec_d < vecEndUnrolled; vec_d += 4)
        {
            ifmCoords0[width] = d + vec_d;
            ifmCoords1[width] = d + vec_d + 1;
            ifmCoords2[width] = d + vec_d + 2;
            ifmCoords3[width] = d + vec_d + 3;

            // load index value from indices tensor
            int index0 = s_i32_ld_g((__global__ int*)idxAddr0);
            int index1 = s_i32_ld_g((__global__ int*)idxAddr1);
            int index2 = s_i32_ld_g((__global__ int*)idxAddr2);
            int index3 = s_i32_ld_g((__global__ int*)idxAddr3);

            idxAddr0 = ( __global__ int*)gen_addr(idxCoords0, indices); idxCoords0[depth] += 4;
            idxAddr1 = ( __global__ int*)gen_addr(idxCoords1, indices); idxCoords1[depth] += 4;
            idxAddr2 = ( __global__ int*)gen_addr(idxCoords2, indices); idxCoords2[depth] += 4;
            idxAddr3 = ( __global__ int*)gen_addr(idxCoords3, indices); idxCoords3[depth] += 4;

            // overwrite the coords value
            dataCoords0[width] = index0;
            dataCoords1[width] = index1;
            dataCoords2[width] = index2;
            dataCoords3[width] = index3;

            for(int dd = inputDepthStart; dd < inputDepthEnd; dd += inputDepthStep)
            {
                dataCoords0[depth] = dd; ifmCoords0[depth] = dd;
                dataCoords1[depth] = dd; ifmCoords1[depth] = dd;
                dataCoords2[depth] = dd; ifmCoords2[depth] = dd;
                dataCoords3[depth] = dd; ifmCoords3[depth] = dd;

                x0 = v_bf16_ld_tnsr_b(ifmCoords0, input);
                x1 = v_bf16_ld_tnsr_b(ifmCoords1, input);
                x2 = v_bf16_ld_tnsr_b(ifmCoords2, input);
                x3 = v_bf16_ld_tnsr_b(ifmCoords3, input);

                v_bf16_st_tnsr(dataCoords0, output, x0);
                v_bf16_st_tnsr(dataCoords1, output, x1);
                v_bf16_st_tnsr(dataCoords2, output, x2);
                v_bf16_st_tnsr(dataCoords3, output, x3);
            }
        }

        // process tail cases
        int ld0en = vecTail > 0;
        int ld1en = vecTail > 1;
        int ld2en = vecTail > 2;

        // load index value from indices tensor
        int index0 = s_i32_ld_g((__global__ int*)idxAddr0, 0, 0, ld0en);
        int index1 = s_i32_ld_g((__global__ int*)idxAddr1, 0, 0, ld1en);
        int index2 = s_i32_ld_g((__global__ int*)idxAddr2, 0, 0, ld2en);

        ifmCoords0[width] = d + vecEndUnrolled;
        ifmCoords1[width] = d + vecEndUnrolled + 1;
        ifmCoords2[width] = d + vecEndUnrolled + 2;

        dataCoords0[width] = index0;
        dataCoords1[width] = index1;
        dataCoords2[width] = index2;

        for(int dd = inputDepthStart; dd < inputDepthEnd; dd += inputDepthStep)
        {
            dataCoords0[depth] = dd; ifmCoords0[depth] = dd;
            dataCoords1[depth] = dd; ifmCoords1[depth] = dd;
            dataCoords2[depth] = dd; ifmCoords2[depth] = dd;

            x0 = v_bf16_ld_tnsr_b(ifmCoords0, input, 0, 0, ld0en);
            x1 = v_bf16_ld_tnsr_b(ifmCoords1, input, 0, 0, ld1en);
            x2 = v_bf16_ld_tnsr_b(ifmCoords2, input, 0, 0, ld2en);
            
            v_bf16_st_tnsr(dataCoords0, output, x0, 0, ld0en);
            v_bf16_st_tnsr(dataCoords1, output, x1, 0, ld1en);
            v_bf16_st_tnsr(dataCoords2, output, x2, 0, ld2en);
        }
    }
}