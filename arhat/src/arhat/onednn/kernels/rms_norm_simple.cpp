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

const char g_kernelCodeRmsNorm[] = R"(
__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void rms_norm_simple(
        const global float *src0,
        global float *dst,
        float eps) {
    src0 += SRC0_BASE;
    dst += DST_BASE;

    int i0 = get_group_id(2);
    int i1 = get_group_id(1);
    int i2 = get_group_id(0);

    const global float *x_scalar = src0 + i0 * SRC0_S0 + i1 * SRC0_S1 + i2 * SRC0_S2;
    const global float4 *x = (const global float4 *)x_scalar;

    float4 sumf = 0;
    float all_sum = 0;

    local float sum[SG_NUM];

    // parallel sum
    for (int i3 = get_local_id(0); i3 < SRC0_D3 / 4; i3 += get_local_size(0)) {
        sumf += x[i3] * x[i3];
    }
    all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    all_sum = sub_group_reduce_add(all_sum);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = all_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // broadcast
    for (int i = get_local_size(0) / SG_SIZE / 2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_local_id(0) == 0) {
        for (int i = 4 * (SRC0_D3 / 4); i < SRC0_D3; i++) {
            sum[0] += x_scalar[i] * x_scalar[i];
        }
        sum[0] /= (float)SRC0_D3;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const float mean = sum[0];
    const float scale = rsqrt(mean + eps);

    global float *y_scalar = dst + i0 * DST_S0 + i1 * DST_S1 + i2 * DST_S2;
    global float4 *y = (global float4 *)y_scalar;

    for (int i3 = get_local_id(0); i3 < SRC0_D3 / 4; i3 += get_local_size(0)) {
        y[i3] = x[i3] * scale;
    }
    if (get_local_id(0) == 0) {
        for (int i3 = 4 * (SRC0_D3 / 4); i3 < SRC0_D3; i3++) {
            y_scalar[i3] = x_scalar[i3] * scale;
        }
    }
} 

)";

} // namespace

const char *RmsNormSimpleKernelCode() {
    return g_kernelCodeRmsNorm;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

