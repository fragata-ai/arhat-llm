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

const char g_codeAddOp[] = R"(
inline float op_add(const float a, const float b) {
    return a + b;
}

)";

const char g_codeSubOp[] = R"(
inline float op_sub(const float a, const float b) {
    return a - b;
}

)";

const char g_codeMulOp[] = R"(
inline float op_mul(const float a, const float b) {
    return a * b;
}

)";

const char g_codeDivOp[] = R"(
inline float op_div(const float a, const float b) {
    return a / b;
}

)";

const char g_kernelCodeBinary[] = R"(
#define FASTDIV(a, b) fastdiv(a, b##_fd0, b##_fd1)
#define FASTMOD(a, b) fastmod(a, b, b##_fd0, b##_fd1)

kernel void binary(
        const global SRC0_T *src0,
        const global SRC1_T *src1,
        global DST_T *dst
        SHAPE_INFO_ARGS) {

    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    dst += DST_BASE;

    const uint i3s = LDIM_0 * GID_0 + LID_0;
    const uint i2 = LDIM_1 * GID_1 + LID_1;
    const uint i1 = FASTDIV(LDIM_2 * GID_2 + LID_2, DST_D0);
    const uint i0 = LDIM_2 * GID_2 + LID_2 - i1 * DST_D0;

    if (i3s >= (uint)DST_D3 || i2 >= (uint)DST_D2 || i1 >= (uint)DST_D1 || i0 >= DST_D0) {
        return;
    }

    const uint i12 = FASTMOD(i2, SRC1_D2);
    const uint i11 = FASTMOD(i1, SRC1_D1);
    const uint i10 = FASTMOD(i0, SRC1_D0);

    const int i_src0 = i0 * SRC0_S0 + i1 * SRC0_S1 + i2 * SRC0_S2;
    const int i_src1 = i10 * SRC1_S0 + i11 * SRC1_S1 + i12 * SRC1_S2;
    const int i_dst = i0 * DST_S0 + i1 * DST_S1 + i2 * DST_S2;

    const global SRC0_T *src0_row = src0 + i_src0;
    global DST_T *dst_row = dst + i_dst;

    for (int i3 = i3s; i3 < DST_D3; i3 += LDIM_0 * GDIM_0) {
        const uint i13 = FASTMOD(i3, SRC1_D3);

        float result = (float)src0_row[i3 * SRC0_S3];
        result = BIN_OP(result, (float)src1[i_src1 + i13 * SRC1_S3]);

        dst_row[i3] = (DST_T)result;
    }
}

)";

const char g_kernelCodeBinaryUnravel[] = R"(
#define FASTDIV(a, b) fastdiv(a, b##_fd0, b##_fd1)
#define FASTMOD(a, b) fastmod(a, b, b##_fd0, b##_fd1)

kernel void binary(
        const global SRC0_T *src0,
        const global SRC1_T *src1,
        global DST_T *dst
        SHAPE_INFO_ARGS) {

    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    dst += DST_BASE;

    const int i = LDIM_0 * GID_0 + LID_0;

    const uint i0 = FASTDIV(i, DST_D123);
    const uint i1 = FASTDIV(i - i0 * DST_D123, DST_D23);
    const uint i2 = FASTDIV(i - i0 * DST_D123 - i1 * DST_D23, DST_D3);
    const uint i3 = i - i0 * DST_D123 - i1 * DST_D23 - i2 * DST_D3;

    if (i3 >= DST_D3 || i2 >= DST_D2 || i1 >= DST_D1 || i0 >= DST_D0) {
        return;
    }

    const int i12 = FASTMOD(i2, SRC1_D2);
    const int i11 = FASTMOD(i1, SRC1_D1);
    const int i10 = FASTMOD(i0, SRC1_D0);

    const int i_src0 = i0 * SRC0_S0 + i1 * SRC0_S1 + i2 * SRC0_S2;
    const int i_src1 = i10 * SRC1_S0 + i11 * SRC1_S1 + i12 * SRC1_S2;
    const int i_dst = i0 * DST_S0 + i1 * DST_S1 + i2 * DST_S2;

    const global SRC0_T *src0_row = src0 + i_src0;
    global DST_T *dst_row = dst + i_dst;

    const int i13 = FASTMOD(i3, SRC1_D3);

    float result = (float)src0_row[i3 * SRC0_S3];
    result = BIN_OP(result, (float)src1[i_src1 + i13 * SRC1_S3]);

    dst_row[i3] = (DST_T)result;
}

)";

} // namespace

const char *BinarySimpleAddOpCode() {
    return g_codeAddOp;
}

const char *BinarySimpleSubOpCode() {
    return g_codeSubOp;
}

const char *BinarySimpleMulOpCode() {
    return g_codeMulOp;
}

const char *BinarySimpleDivOpCode() {
    return g_codeDivOp;
}

const char *BinarySimpleKernelCode() {
    return g_kernelCodeBinary;
}

const char *BinarySimpleUnravelKernelCode() {
    return g_kernelCodeBinaryUnravel;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

