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

const char g_kernelCodeNorm[] = R"(
kernel void norm_simple(
        const global float *src,
        global float *dst,
        float eps) {
    src += SRC_BASE;
    dst += DST_BASE;

    int i0 = get_group_id(2);
    int i1 = get_group_id(1);
    int i2 = get_group_id(0);

    const global float *x = src + i0 * SRC_S0 + i1 * SRC_S1 + i2 * SRC_S2;

    local float sum[LWS0];

    // MEAN
    // parallel sum
    sum[get_local_id(0)] = 0.0f;
    for (int i3 = get_local_id(0); i3 < SRC_D3; i3 += get_local_size(0)) {
        sum[get_local_id(0)] += x[i3];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // reduce
    for (uint i = get_local_size(0) / 2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean = sum[0] / (float)SRC_D3;

    barrier(CLK_LOCAL_MEM_FENCE);

    // recenter and VARIANCE
    global float *y = dst + i0 * DST_S0 + i1 * DST_S1 + i2 * DST_S2;
    sum[get_local_id(0)] = 0.0f;
    for (int i3 = get_local_id(0); i3 < SRC_D3; i3 += get_local_size(0)) {
        y[i3] = x[i3] - mean;
        sum[get_local_id(0)] += y[i3] * y[i3];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // reduce
    for (uint i = get_local_size(0) / 2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float variance = sum[0] / (float)SRC_D3;

    float scale = rsqrt(variance + eps);
    for (int i3 = get_local_id(0); i3 < SRC_D3; i3 += get_local_size(0)) {
        y[i3] = y[i3] * scale;
    }
} 

)";

} // namespace

const char *NormSimpleKernelCode() {
    return g_kernelCodeNorm;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

