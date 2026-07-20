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

const char g_kernelCodeQuantizeVec_Q8_1[] = R"(
#define QK8_1 32

typedef struct {
    half2 ds;       // [delta, d * sum(qs[*])]
    char qs[QK8_1]; // quants
} block_q8_1; 

#define FASTDIV(a, b) fastdiv(a, b##_fd0, b##_fd1)

__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void quantize_vec(
        const global float *x, 
        global char *vy
        SHAPE_INFO_ARGS) {

    x += SRC_BASE;
    vy += DST_BASE;

    const long i3 = (long)LDIM_0 * GID_0 + LID_0;

    if (i3 >= DST_D3) {
        return;
    }

    const long i0 = FASTDIV(GID_2, DST_D1);
    const long i1 = GID_2 - i0 * DST_D1;
    const long i2 = GID_1;

    const long i_cont = ((i0 * DST_D1 + i1) * DST_D2 + i2) * DST_D3 + i3;

    global block_q8_1 *y = (global block_q8_1 *)vy;

    const long ib = i_cont / QK8_1;  // block index
    const long iqs = i_cont % QK8_1; // quant index

    const float xi = (i3 < SRC_D3) ? x[i0 * SRC_S0 + i1 * SRC_S1 + i2 * SRC_S2 + i3] : 0.0f;
    float amax = fabs(xi);
    float sum = xi;

    // require SG_SIZE = QK8_1 (= 32)
    amax = sub_group_reduce_max(amax);
    sum = sub_group_reduce_add(sum);

    const float d = amax / 127.0f;
    const char q = (amax == 0.0f) ? 0 : round(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = (half2)(d, sum);
}

)";

const char g_kernelCodeQuantizeVec_Q8_1_Soa[] = R"(
#define QK8_1 32

#define FASTDIV(a, b) fastdiv(a, b##_fd0, b##_fd1)

__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void quantize_vec(
        const global float *x, 
        global char *vy
        SHAPE_INFO_ARGS) {

    x += SRC_BASE;
    vy += DST_BASE;

    const long i3 = (long)LDIM_0 * GID_0 + LID_0;

    if (i3 >= DST_D3) {
        return;
    }

    const long i0 = FASTDIV(GID_2, DST_D1);
    const long i1 = GID_2 - i0 * DST_D1;
    const long i2 = GID_1;

    const float xi = (i3 < SRC_D3) ? x[i0 * SRC_S0 + i1 * SRC_S1 + i2 * SRC_S2 + i3] : 0.0f;
    float amax = fabs(xi);
    float sum = xi;

    // require SG_SIZE = QK8_1 (= 32)
    amax = sub_group_reduce_max(amax);
    sum = sub_group_reduce_add(sum);

    const float d = amax / 127.0f;
    const char q = (amax == 0.0f) ? 0 : round(xi / d);

    const long row_offset = i0 * DST_S0 + i1 * DST_S1 + i2 * DST_S2;

    vy[row_offset + i3] = q;

    global half2 *ds = (global half2 *)(vy + row_offset + DST_D3) + i3 / QK8_1;
    if (i3 % QK8_1 == 0) {
        *ds = (half2)(d, sum);
    }
}

)";

} // namespace

const char *QuantizeVec_Q8_1_KernelCode() {
    return g_kernelCodeQuantizeVec_Q8_1;
}

const char *QuantizeVec_Q8_1_Soa_KernelCode() {
    return g_kernelCodeQuantizeVec_Q8_1_Soa;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

