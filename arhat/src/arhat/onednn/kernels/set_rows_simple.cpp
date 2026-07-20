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

const char g_kernelCodeSetRows[] = R"(
kernel void set_rows_simple(
        const global SRC0_TYPE *src0,
        const global SRC1_TYPE *src1,
        global DST_TYPE *dst
        SHAPE_INFO_ARGS) {

    int i0 = GID_2;
    int i1 = GID_1;
    int i2 = GID_0 * LDIM_1 + LID_1;

    if (i2 >= SRC0_D2) {
        return;
    }

    int k1 = i0 % SRC1_D1;
    int k2 = i1 % SRC1_D2;

    SRC1_TYPE r = src1[SRC1_BASE + k1 * SRC1_S1 + k2 * SRC1_S2 + i2 * SRC1_S3];

    const global SRC0_TYPE *src_row = src0 + SRC0_BASE + i0 * SRC0_S0 + i1 * SRC0_S1 + i2 * SRC0_S2;
    global DST_TYPE *dst_row = dst + DST_BASE + i0 * DST_S0 + i1 * DST_S1 + r * DST_S2;

    for (int ind = LID_0; ind < DST_D3; ind += LDIM_0) {
        dst_row[ind] = (DST_TYPE)src_row[ind];
    }
}

)";

} // namespace

const char *SetRowsSimpleKernelCode() {
    return g_kernelCodeSetRows;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

