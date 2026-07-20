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

const char g_kernelCodeGetRows[] = R"(
#define FASTDIV(a, b) fastdiv(a, b##_fd0, b##_fd1)

kernel void get_rows_simple(
        const global SRC0_TYPE *src0, 
        const global int *src1, 
        global DST_TYPE *dst
        SHAPE_INFO_ARGS) {

    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    dst += DST_BASE;

    for (long z = GID_2; z < SRC1_D1 * SRC1_D2; z += GDIM_2) {
        for (long i03 = GID_1 * LDIM_0 + LID_0; i03 < SRC0_D3; i03 += GDIM_1 * LDIM_0) {
            const int i13 = GID_0;
            const long i12 = FASTDIV((uint)z, SRC1_D1);
            const long i11 = z - i12 * SRC1_D1;

            if (i03 >= SRC0_D3) {
                return;
            }

            const int i02 = src1[i11 * SRC1_S1 + i12 * SRC1_S2 + i13 * SRC1_S3];

            global DST_TYPE *dst_row = dst + i11 * DST_S0 + i12 * DST_S1 + i13 * DST_S2;
            const global SRC0_TYPE *src0_row = src0 + i11 * SRC0_S0 + i12 * SRC0_S1 + i02 * SRC0_S2;

            dst_row[i03] = (DST_TYPE)src0_row[i03];
        }
    }
}

)";

} // namespace

const char *GetRowsSimpleKernelCode() {
    return g_kernelCodeGetRows;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

