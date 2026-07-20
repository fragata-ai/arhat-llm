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

const char g_kernelCodeCpy[] = R"(
kernel void cpy_simple(const global SRC_TYPE *src, global DST_TYPE *dst) {
    src += SRC_BASE;
    dst += DST_BASE;

    int is0 = get_group_id(2);
    int is1 = get_group_id(1);
    int is2 = get_group_id(0);

    int n = is0 * SRC_D1 * SRC_D2 * SRC_D3 + is1 * SRC_D2 * SRC_D3 + is2 * SRC_D3;

    int id0 = n / (DST_D1 * DST_D2 * DST_D3);
    n -= id0 * DST_D1 * DST_D2 * DST_D3;
    int id1 = n / (DST_D2 * DST_D3);
    n -= id1 * DST_D2 * DST_D3;
    int id2 = n / DST_D3;
    n -= id2 * DST_D3;
    int id3 = n;

    const global SRC_TYPE *src_ptr = src + is0 * SRC_S0 + is1 * SRC_S1 + is2 * SRC_S2;
    global DST_TYPE *dst_ptr = dst + id0 * DST_S0 + id1 * DST_S1 + id2 * DST_S2 + id3 * DST_S3;

    for (int is3 = get_local_id(0); is3 < SRC_D3; is3 += get_local_size(0)) {
        dst_ptr[is3 * DST_S3] = (DST_TYPE)src_ptr[is3 * SRC_S3];
    }
} 

)";

} // namespace

const char *CpySimpleKernelCode() {
    return g_kernelCodeCpy;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

