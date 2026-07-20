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

const char g_kernelCodeMulMat[] = R"(
#define FASTDIV(a, b) fastdiv(a, b##_fd0, b##_fd1)
#define FASTMOD(a, b) fastmod(a, b, b##_fd0, b##_fd1)

#define MAD(acc, x, y) acc = mad(x, y, acc)

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
kernel void mul_mat_vec(
        const global T *x, 
        const global float *y, 
        const global int *ids, 
        global float *dst,
        const int ncols2, 
        const uint nchannels_y, 
        const uint nchannels_y_fd0, 
        const uint nchannels_y_fd1, 
        const int stride_row, 
        const int stride_col_y2, 
        const int stride_col_dst,
        const uint channel_ratio, 
        const uint channel_ratio_fd0, 
        const uint channel_ratio_fd1, 
        const int stride_channel_x, 
        const int stride_channel_y, 
        const int stride_channel_dst,
        const uint sample_ratio, 
        const uint sample_ratio_fd0, 
        const uint sample_ratio_fd1, 
        const int stride_sample_x, 
        const int stride_sample_y, 
        const int stride_sample_dst,
        const int ids_stride
        SHAPE_INFO_ARGS) {

    x += SRC0_BASE;
    y += SRC1_BASE;
    if (ids != NULL) {
        ids += SRC2_BASE;
    }
    dst += DST_BASE;
    
    // for MUL_MAT_ID: GID_1 = n_expert_used, GID_2 = NCOLS_DST (tokens)
    const int row = GID_0;
    const int channel_dst = GID_1;
    const int tid = LID_0;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    if (IS_MULTI_TOKEN_ID) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path
        // causes a perf regression for n_tokens = 1 case
        token_idx = GID_2;
        channel_x = ids[channel_dst + token_idx * ids_stride];
        channel_y = FASTMOD(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        token_idx = ids ? GID_2 : 0;
        channel_x = 
            ids ? 
                ids[GID_1 + token_idx * ids_stride] : 
                FASTDIV((uint) channel_dst, channel_ratio);
        channel_y = ids ? FASTMOD(GID_1, nchannels_y) : channel_dst;
        sample_dst = ids ? 0 : GID_2;
    }

    const int sample_x = FASTDIV((uint)sample_dst, sample_ratio);
    const int sample_y = sample_dst;

    x += (long)sample_x * stride_sample_x + channel_x * stride_channel_x + row * stride_row;
    y += (long)sample_y * stride_sample_y + channel_y * stride_channel_y;
    dst += (long)sample_dst * stride_sample_dst + channel_dst * stride_channel_dst;
    if (IS_MULTI_TOKEN_ID) {
        y += token_idx * stride_col_y2 * 2;
        dst += token_idx * stride_col_dst;
    }

    const global float2 *y2 = (const global float2 *)y;

    local float local_mem[NE_LOCAL];
    local float *buf_iw = local_mem;

    if (BLOCK_SIZE > SG_SIZE) {
        if (tid < SG_SIZE) {
            buf_iw[tid] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float sumf[NCOLS_DST] = {0.0f};

#if defined(T_IS_FLOAT)
    const global float2 *x2 = (const global float2 *)x;

    for (int col2 = tid; col2 < ncols2; col2 += BLOCK_SIZE) {
        const float2 tmpx = x2[col2];

        unroll_for (int j = 0; j < NCOLS_DST; j++) {
            const float2 tmpy = y2[j * stride_col_y2 + col2];
            MAD(sumf[j], tmpx.x, tmpy.x);
            MAD(sumf[j], tmpx.y, tmpy.y);
        }
    }

#elif defined(T_IS_HALF)
    const global half2 *x2 = (const global half2 *)x;

#if defined(TYPE_ACC_IS_FLOAT)
    for (int col2 = tid; col2 < ncols2; col2 += BLOCK_SIZE) {
        const float2 tmpx = convert_float2(x2[col2]);

        unroll_for (int j = 0; j < NCOLS_DST; j++) {
            const float2 tmpy = y2[j * stride_col_y2 + col2];
            MAD(sumf[j], tmpx.x, tmpy.x);
            MAD(sumf[j], tmpx.y, tmpy.y);
        }
    }

#elif defined(TYPE_ACC_IS_HALF)
    half2 sumh2[NCOLS_DST] = {(half2)(0.0f, 0.0f)};

    for (int col2 = tid; col2 < ncols2; col2 += BLOCK_SIZE) {
        const half2 tmpx = x2[col2];

        unroll_for (int j = 0; j < NCOLS_DST; j++) {
            const float2 tmpy = y2[j * stride_col_y2 + col2];
            sumh2[j] += tmpx * (half2)(tmpy.x, tmpy.y);
        }
    }

    unroll_for (int j = 0; j < NCOLS_DST; j++) {
        sumf[j] = convert_float(sumh2[j].x) + convert_float(sumh2[j].y);
    }

#else
#error "Invalid ACC_TYPE"
#endif // defined(ACC_TYPE_IS_*)

// SKIPPED: defined(T_IS_BFLOAT16)
#else
#error "Invalid T"
#endif // defined(T_IS_*)

    unroll_for (int j = 0; j < NCOLS_DST; j++) {
        sumf[j] = sub_group_reduce_add(sumf[j]);

        if (BLOCK_SIZE > SG_SIZE) {
            buf_iw[tid / SG_SIZE] = sumf[j];
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tid < SG_SIZE) {
                sumf[j] = buf_iw[tid];
                sumf[j] = sub_group_reduce_add(sumf[j]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (tid >= NCOLS_DST) {
        return;
    }

    float value = sumf[tid];

    dst[tid * stride_col_dst + row] = value;
}

)";

} // namespace

const char *MulMatVecKernelCode() {
    return g_kernelCodeMulMat;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

