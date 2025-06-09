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

// #pragma tpc_printf (enable)

// #define print_vec_int(vec, str)  { printf(str); for (int ii=0; ii<64; ii++) {  printf("%d, ", vec[ii]);  }; printf("\n"); }
// #define print_vec_float(vec, str)  { printf(str); for (int ii=0; ii<64; ii++) {  printf("%f, ", vec[ii]);  }; printf("\n"); }

// space left for register spill
#define REG_SPILL_REDUCTION 20
// space assigned for data. as we do not have LUT, out total space is 320 256-byte vectors
#define VLM_MAX_VECTOR (320 - REG_SPILL_REDUCTION)

//local memory definition
__local__ float64 vlm[VLM_MAX_VECTOR];

void main(tensor weight, tensor indices, tensor offset, tensor output, int n_tpc)
{
    const int depth   = 0;
    const int width   = 1;
    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd   = get_index_space_size() + indexSpaceStart;

    const int depthStep   = 64;
    
    // Get the number of indices
    const int offsetSize = get_dim_size(offset, 0); // this is a batchsize + 1
    const int weightDepthSize = get_dim_size(weight, 0);
    
    // Use ceiling division to ensure all elements are divided across PEs
    int unit = (offsetSize - 1 + n_tpc - 1) / n_tpc;

    const int OffsetDepthStart  = indexSpaceStart[depth] * unit;
    const int OffsetDepthEnd    = s_i32_min(indexSpaceEnd[depth] * unit, offsetSize - 1);  // Ensure not to exceed indices size
    
    // printf("offset start: %d\n", OffsetDepthStart);
    // printf("offset end: %d\n", OffsetDepthEnd);

    const int weightDepthStart  = 0;
    const int weightDepthEnd  = weightDepthSize;
    const int vlm_num_vectors = (weightDepthSize + depthStep - 1) / depthStep;

    int5 outCoords = {0};
    
    int5 offsetCoords0 = {0};
    int5 offsetCoords1 = {0};

    int5 indexCoords0 = {0};
    int5 indexCoords1 = {0};
    int5 indexCoords2 = {0};
    int5 indexCoords3 = {0};

    int5 weightCoords0 = {0};
    int5 weightCoords1 = {0};
    int5 weightCoords2 = {0};
    int5 weightCoords3 = {0};

    float64 x0, x1, x2, x3;

    #pragma loop_taken
    for (int d = OffsetDepthStart; d < OffsetDepthEnd; d += 1)
    {
        offsetCoords0[depth] = d;
        offsetCoords1[depth] = d + 1;
        outCoords[width] = d;

        __global__ int* offsetAddr0 = ( __global__ int*)gen_addr(offsetCoords0, offset);
        __global__ int* offsetAddr1 = ( __global__ int*)gen_addr(offsetCoords1, offset);

        // load offset value from offset tensor
        int idxStart = s_i32_ld_g((__global__ int*)offsetAddr0);
        int idxEnd = s_i32_ld_g((__global__ int*)offsetAddr1);
        
        // printf("idxStart: %d\n", idxStart);
        // printf("idxEnd: %d\n", idxEnd);
                
        int vecEnd = idxEnd-idxStart;
        int vecTail = vecEnd & 3;
        int vecEndUnrolled = vecEnd - vecTail;

        // process tail cases
        int ld0en = vecTail > 0;
        int ld1en = vecTail > 1;
        int ld2en = vecTail > 2;

        #pragma loop_taken
        for (int k = 0; k < vlm_num_vectors; k++)
        {
            vlm[k] = 0;
        }

        // idx 
        indexCoords0[depth] = idxStart;
        indexCoords1[depth] = idxStart + 1;
        indexCoords2[depth] = idxStart + 2;
        indexCoords3[depth] = idxStart + 3;

        __global__ int* indexAddr0 = ( __global__ int*)gen_addr(indexCoords0, indices); indexCoords0[depth] += 4;
        __global__ int* indexAddr1 = ( __global__ int*)gen_addr(indexCoords1, indices); indexCoords1[depth] += 4;
        __global__ int* indexAddr2 = ( __global__ int*)gen_addr(indexCoords2, indices); indexCoords2[depth] += 4;
        __global__ int* indexAddr3 = ( __global__ int*)gen_addr(indexCoords3, indices); indexCoords3[depth] += 4;

        for (int vec_d = 0; vec_d < vecEndUnrolled; vec_d += 4)
        {
            // load target weight index value from indices tensor
            int weight_index0 = s_i32_ld_g((__global__ int*)indexAddr0); 
            int weight_index1 = s_i32_ld_g((__global__ int*)indexAddr1); 
            int weight_index2 = s_i32_ld_g((__global__ int*)indexAddr2); 
            int weight_index3 = s_i32_ld_g((__global__ int*)indexAddr3); 

            indexAddr0 = ( __global__ int*)gen_addr(indexCoords0, indices); indexCoords0[depth] += 4;
            indexAddr1 = ( __global__ int*)gen_addr(indexCoords1, indices); indexCoords1[depth] += 4;
            indexAddr2 = ( __global__ int*)gen_addr(indexCoords2, indices); indexCoords2[depth] += 4;
            indexAddr3 = ( __global__ int*)gen_addr(indexCoords3, indices); indexCoords3[depth] += 4;

            weightCoords0[width] = weight_index0;
            weightCoords1[width] = weight_index1;
            weightCoords2[width] = weight_index2;
            weightCoords3[width] = weight_index3;

            for(int dd = weightDepthStart, i = 0; dd < weightDepthEnd; dd += depthStep, i++)
            {
                weightCoords0[depth] = dd;
                weightCoords1[depth] = dd;
                weightCoords2[depth] = dd;
                weightCoords3[depth] = dd;

                x0 = v_f32_ld_tnsr_b(weightCoords0, weight);
                x1 = v_f32_ld_tnsr_b(weightCoords1, weight);
                x2 = v_f32_ld_tnsr_b(weightCoords2, weight);
                x3 = v_f32_ld_tnsr_b(weightCoords3, weight);

                vlm[i] = v_f32_add_b(x0, vlm[i]);
                vlm[i] = v_f32_add_b(x1, vlm[i]);
                vlm[i] = v_f32_add_b(x2, vlm[i]);
                vlm[i] = v_f32_add_b(x3, vlm[i]);
            }
        }
        // load target weight index value from indices tensor
        int weight_index0 = s_i32_ld_g((__global__ int*)indexAddr0, 0, 0, ld0en);
        int weight_index1 = s_i32_ld_g((__global__ int*)indexAddr1, 0, 0, ld1en);
        int weight_index2 = s_i32_ld_g((__global__ int*)indexAddr2, 0, 0, ld2en);
        
        // printf("weight_index0: %d\n", weight_index0);
        // printf("weight_index1: %d\n", weight_index1);
        // printf("weight_index2: %d\n", weight_index2);
        
        weightCoords0[width] = weight_index0;
        weightCoords1[width] = weight_index1;
        weightCoords2[width] = weight_index2;

        for(int dd = weightDepthStart, i = 0; dd < weightDepthEnd; dd += depthStep, i++)
        {
            weightCoords0[depth] = dd;
            weightCoords1[depth] = dd;
            weightCoords2[depth] = dd;
            weightCoords3[depth] = dd;
            outCoords[depth] = dd;
            
            x0 = v_f32_ld_tnsr_b(weightCoords0, weight, 0, 0, ld0en);
            x1 = v_f32_ld_tnsr_b(weightCoords1, weight, 0, 0, ld1en);
            x2 = v_f32_ld_tnsr_b(weightCoords2, weight, 0, 0, ld2en);

            vlm[i] = v_f32_add_b(x0, vlm[i], 0, vlm[i], ld0en);
            vlm[i] = v_f32_add_b(x1, vlm[i], 0, vlm[i], ld1en);
            vlm[i] = v_f32_add_b(x2, vlm[i], 0, vlm[i], ld2en);

            v_f32_st_tnsr(outCoords, output, vlm[i]);
        }
    }
}

