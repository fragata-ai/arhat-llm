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

const char g_kernelCodeGroupNorm[] = R"(
__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void group_norm_simple(
        const global float *src,
        global float *dst,
        float eps) {
    const int N = SRC_D1 * SRC_D2 * SRC_D3;

    int start = get_group_id(0) * GROUP_SIZE;
    int end = start + GROUP_SIZE;

    start += get_local_id(0);
    if (end >= N) {
        end = N;
    }

    float tmp = 0.0f;

    for (int j = start; j < end; j += get_local_size(0)) {
        tmp += src[j];
    }

    tmp = sub_group_reduce_add(tmp);

    const float mean = tmp / (float)GROUP_SIZE;
    tmp = 0.0f;

    for (int j = start; j < end; j += get_local_size(0)) {
        float xi = src[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = sub_group_reduce_add(tmp);

    const float variance = tmp / (float)GROUP_SIZE;
    const float scale = rsqrt(variance + eps);
    for (int j = start; j < end; j += get_local_size(0)) {
        dst[j] *= scale;
    }
} 

)";

} // namespace

const char *GroupNormSimpleKernelCode() {
    return g_kernelCodeGroupNorm;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

