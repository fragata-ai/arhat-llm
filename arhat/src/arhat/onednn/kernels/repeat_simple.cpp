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

const char g_kernelCodeRepeat[] = R"(
kernel void repeat_simple(const global float *src, global float *dst) {
    src += SRC_BASE;
    dst += DST_BASE;

    const int id0 = get_group_id(2);
    const int id1 = get_group_id(1);
    const int id2 = get_group_id(0);

    const int is0 = id0 % SRC_D0;
    const int is1 = id1 % SRC_D1;
    const int is2 = id2 % SRC_D2;

    const global float *src_ptr = src + is0 * SRC_S0 + is1 * SRC_S1 + is2 * SRC_S2;
    global float *dst_ptr = dst + id0 * DST_S0 + id1 * DST_S1 + id2 * DST_S2;

    for (int id3 = get_local_id(0); id3 < DST_D3; id3 += get_local_size(0)) {
        const int is3 = id3 % SRC_D3;
        dst_ptr[id3 * DST_S3] = src_ptr[is3 * SRC_S3];
    } 
}

)";

} // namespace

const char *RepeatSimpleKernelCode() {
    return g_kernelCodeRepeat;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

