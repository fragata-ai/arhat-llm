/* 
* MIT License
*
* Copyright (c) 2026 FRAGATA COMPUTER SYSTEMS AG
* Copyright (c) 2023-2026 The ggml authors
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "arhat/onednn/kernels/code.hpp"

namespace arhat {
namespace onednn {
namespace kernels {

namespace {

const char g_kernelCodeSolveTri[] = R"(
kernel void solve_tri_simple(
        const global float *src0,
        const global float *src1,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    dst += DST_BASE;

    int col = get_global_id(0);
    int i1 = get_global_id(1);
    int i0 = get_global_id(2);

    if (i0 >= DST_D0 || i1 >= DST_D1 || col >= DST_D3) {
        return;
    }

    const global float *lb = src0 + i0 * SRC0_S0 + i1 * SRC0_S1;
    const global float *bb = src1 + i0 * SRC1_S0 + i1 * SRC1_S1;
    global float *xb = dst + i0 * DST_S0 + i1 * DST_S1;

    for (int row = 0; row < N; row++) {
        const global float *pb = bb + row * SRC1_S2 + col * SRC1_S3;

        float sum = 0.0f;
        for (int j = 0; j < row; j++){
            const global float *pl = lb + row * SRC0_S2 + j * SRC0_S3;
            const global float *px = xb + j * DST_S2 + col * DST_S3;
            sum += (*pl) * (*px);
        }

        const global float *pdiag = lb + row * SRC0_S2 + row * SRC0_S3;
        global float *pout = xb + row * DST_S2 + col * DST_S3;

        *pout = ((*pb) - sum) / (*pdiag);
    }
} 

)";

} // namespace

const char *SolveTriSimpleKernelCode() {
    return g_kernelCodeSolveTri;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

