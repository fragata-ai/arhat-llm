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

// [add_id.cl]
const char g_kernelCodeAddId[] = R"(
kernel void add_id_simple(
        const global float *src0,
        const global float *src1,
        const global int *src2,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    src2 += SRC2_BASE;
    dst += DST_BASE;

    int i2 = get_group_id(0);
    int i1 = get_group_id(1);

    int k = src2[i1 * SRC2_S2 + i2 * SRC2_S3];

    global float *dst_row  = dst + i1 * DST_S1 + i2 * DST_S2;
    const global float *src0_row = src0 + i1 * SRC0_S1 + i2 * SRC0_S2;
    const global float *src1_row = src1 + k * SRC1_S2;

    for (int i3 = get_local_id(0); i3 < DST_D3; i3 += get_local_size(0)) {
        dst_row[i3] = src0_row[i3] + src1_row[i3];
    }
} 

)";

} // namespace

const char *AddIdSimpleKernelCode() {
    return g_kernelCodeAddId;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

