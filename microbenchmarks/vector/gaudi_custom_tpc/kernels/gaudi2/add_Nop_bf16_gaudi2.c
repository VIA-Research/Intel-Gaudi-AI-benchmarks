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

void main(tensor input0, tensor input1, int N_op, tensor output)
{
    const int depth   = 0;
    const int width   = 1;
    const int height  = 2;
    const int batch   = 3;
    const int fifthDim = 4;

    const int5 index_space_start = get_index_space_offset();
    const int5 index_space_end   = get_index_space_size() + index_space_start;

    int5 ifmCoords = {0, 0, 0, 0, 0};
    int5 ofmCoords = {0, 0, 0, 0, 0};

    // DEPTH
    const int depthStep  = 128;
    const int depthStart = index_space_start[depth] * depthStep;
    const int depthEnd   = index_space_end[depth] * depthStep;

    // WIDTH
    const int widthStep  = 4;
    const int widthStart = 0;
    const int widthEnd   = get_dim_size(input0, width);

    // HEIGHT
    const int heightStep  = 1;
    const int heightStart = index_space_start[height] * heightStep;
    const int heightEnd   = index_space_end[height] * heightStep;

    // BATCH
    const int batchStep  = 1;
    const int batchStart = index_space_start[batch];
    const int batchEnd  = index_space_end[batch];

    // fifthDim
    const int fifthDimStep  = 1;
    const int fifthDimStart = index_space_start[fifthDim];
    const int fifthDimtEnd  = index_space_end[fifthDim];

    bfloat128 x00, x10, x01, x11, x02, x12, x03, x13;

    bfloat128 o0, o1, o2, o3;

    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        ifmCoords[depth] = d;
        ofmCoords[depth] = d;

        for (int f = fifthDimStart; f < fifthDimtEnd; f += fifthDimStep)
        {
            ifmCoords[fifthDim] = f;
            ofmCoords[fifthDim] = f;

            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                ifmCoords[batch] = b;
                ofmCoords[batch] = b;

                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    ifmCoords[height] = h;
                    ofmCoords[height] = h;

                    for (int w = widthStart; w < widthEnd; w += widthStep)
                    {
                        ifmCoords[width] = w;
                        ofmCoords[width] = w;

                        x00 = v_bf16_ld_tnsr_b(ifmCoords, input0);
                        x10 = v_bf16_ld_tnsr_b(ifmCoords, input1);
                        ifmCoords[width] += 1;

                        x01 = v_bf16_ld_tnsr_b(ifmCoords, input0);
                        x11 = v_bf16_ld_tnsr_b(ifmCoords, input1);
                        ifmCoords[width] += 1;

                        x02 = v_bf16_ld_tnsr_b(ifmCoords, input0);
                        x12 = v_bf16_ld_tnsr_b(ifmCoords, input1);
                        ifmCoords[width] += 1;
                        
                        x03 = v_bf16_ld_tnsr_b(ifmCoords, input0);
                        x13 = v_bf16_ld_tnsr_b(ifmCoords, input1);

                        o0 = v_bf16_add_b(x00, x10);
                        o1 = v_bf16_add_b(x01, x11);
                        o2 = v_bf16_add_b(x02, x12);
                        o3 = v_bf16_add_b(x03, x13);

                        for (int N = 0; N < N_op - 1; N += 1)
                        {
                            o0 = v_bf16_add_b(o0, x10);
                            o1 = v_bf16_add_b(o1, x11);
                            o2 = v_bf16_add_b(o2, x12);
                            o3 = v_bf16_add_b(o3, x13);
                        }
                        
                        v_bf16_st_tnsr(ofmCoords, output, o0);
                        ofmCoords[width] += 1;

                        v_bf16_st_tnsr(ofmCoords, output, o1);
                        ofmCoords[width] += 1;

                        v_bf16_st_tnsr(ofmCoords, output, o2);
                        ofmCoords[width] += 1;

                        v_bf16_st_tnsr(ofmCoords, output, o3);
                    }
                }
            }
        }
    }
}
