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

const char g_kernelCodeSsmConv[] = R"(
kernel void ssm_conv_simple(
        const global float *src0,
        const global float *src1,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    dst += DST_BASE;

    int i3 = get_global_id(0);
    int i2 = get_global_id(1);
    int i1 = get_global_id(2);

    int nc  = SRC1_D3;

    const global float *s = src0 + i1 * SRC0_S1 + i3 * SRC0_S2 + i2 * SRC0_S3;
    const global float *c = src1 + i3 * SRC1_S2;
    global float *d = dst + i1 * DST_S1 + i2 * DST_S2 + i3 * DST_S3;
 
    float sumf = 0.0f;

    for (int i3 = 0; i3 < nc; i3++) {
        sumf += s[i3] * c[i3];
    }

    d[0] = sumf;
} 

)";

const char g_kernelCodeSsmConvX4[] = R"(
kernel void ssm_conv_simple_x4(
        const global float *src0,
        const global float *src1,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    dst += DST_BASE;

    int i3 = get_global_id(0);
    int i2 = get_global_id(1);
    int i1 = get_global_id(2);

    int nc  = SRC1_D3;

    const global float4 *s = 
        (const global float4 *)(src0 + i1 * SRC0_S1 + i3 * SRC0_S2 + i2 * SRC0_S3);
    const global float4 *c = 
        (const global float4 *)(src1 + i3 * SRC1_S2);
    global float *d = 
        dst + i1 * DST_S1 + i2 * DST_S2 + i3 * DST_S3;
 
    float sumf = 0.0f;

    for (int i3 = 0; i3 < nc / 4; i3++) {
        sumf += dot(s[i3], c[i3]);
    }

    d[0] = sumf;
} 

)";

} // namespace

const char *SsmConvSimpleKernelCode() {
    return g_kernelCodeSsmConv;
}

const char *SsmConvSimpleX4KernelCode() {
    return g_kernelCodeSsmConvX4;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

