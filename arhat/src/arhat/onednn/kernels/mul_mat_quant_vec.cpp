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

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
kernel void mul_mat_vec_q(
        const global char *vx, 
        const global char *vy, 
        const global int *ids, 
        global float *dst,
        const uint ncols_x, 
        const uint nchannels_y, 
        const uint nchannels_y_fd0, 
        const uint nchannels_y_fd1, 
        const uint stride_row_x, 
        const uint stride_col_y,
        const uint stride_col_dst, 
        const uint channel_ratio, 
        const uint channel_ratio_fd0, 
        const uint channel_ratio_fd1, 
        const uint stride_channel_x,
        const uint stride_channel_y, 
        const uint stride_channel_dst, 
        const uint sample_ratio,
        const uint sample_ratio_fd0,
        const uint sample_ratio_fd1,
        const uint stride_sample_x, 
        const uint stride_sample_y, 
        const uint stride_sample_dst,
        const uint ids_stride,
        const uint ncols_y_padded
        SHAPE_INFO_ARGS) {

    vx += SRC0_BASE;
    vy += SRC1_BASE;
    if (ids != NULL) {
        ids += SRC2_BASE;
    }
    dst += DST_BASE;

    const int tid = SG_SIZE * LID_1 + LID_0;
    const int row0 = ROWS_PER_WG * GID_0;
    const int blocks_per_row_x = ncols_x / QK;

    const uint channel_dst = GID_1;

    uint token_idx = 0;
    uint channel_x;
    uint channel_y;
    uint sample_dst;

    if (IS_MULTI_TOKEN_ID) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes
        // a perf regression for n_tokens=1 case
        token_idx = GID_2;
        channel_x = ids[channel_dst + token_idx * ids_stride];
        channel_y = FASTMOD(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        channel_x = (NCOLS_DST == 1 && ids) ? ids[channel_dst] : FASTDIV(channel_dst, channel_ratio);
        channel_y = (NCOLS_DST == 1 && ids) ? FASTMOD(channel_dst, nchannels_y) : channel_dst;
        sample_dst = GID_2;
    }

    const uint sample_x = FASTDIV(sample_dst, sample_ratio);
    const uint sample_y = sample_dst;

    // partial sum for each thread
    float tmp[NCOLS_DST][ROWS_PER_WG] = {{0.0f}};

#if !USE_SOA_Y
    const global block_q8_1 *y = 
        (const global block_q8_1 *)vy + sample_y * stride_sample_y + channel_y * stride_channel_y;
    if (IS_MULTI_TOKEN_ID) {
        y += token_idx * stride_col_y;
    }
    const int kbx_offset = 
        sample_x * stride_sample_x + channel_x * stride_channel_x + row0 * stride_row_x;

    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row_x; kbx += BLOCKS_PER_ITER) {
        // y block index that aligns with kbx
        const int kby = kbx * (QK / QK8_1); 

        // x block quant index when casting the quants to int
        const int kqs = VDR * (tid % (QI / VDR));

        unroll_for (int j = 0; j < NCOLS_DST; j++) {
            unroll_for (int i = 0; i < ROWS_PER_WG; i++) {
                tmp[j][i] += 
                    VEC_DOT_Q(
                        vx, 
                        &y[j * stride_col_y + kby], 
                        kbx_offset + i * stride_row_x + kbx, 
                        kqs);
            }
        }
    }

#else
    const global char *y0 = vy + sample_y * stride_sample_y + channel_y * stride_channel_y;
    if (IS_MULTI_TOKEN_ID) {
        y0 += token_idx * stride_col_y;
    }

    const int kbx_offset = 
        sample_x * stride_sample_x + channel_x * stride_channel_x + row0 * stride_row_x;

    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row_x; kbx += BLOCKS_PER_ITER) {
        // y block index that aligns with kbx
        const int kby = kbx * (QK / QK8_1); 

        // x block quant index when casting the quants to int
        const int kqs = VDR * (tid % (QI / VDR));

        unroll_for (int j = 0; j < NCOLS_DST; j++) {
            global const char *yqs = y0 + j * stride_col_y;
            global const half2 *yds = (global const half2 *)(yqs + ncols_y_padded);

            unroll_for (int i = 0; i < ROWS_PER_WG; i++) {
                tmp[j][i] += 
                    VEC_DOT_Q(
                        vx, 
                        &yqs[kby * QK8_1],
                        &yds[kby],
                        kbx_offset + i * stride_row_x + kbx, 
                        kqs);
            }
        }
    }
#endif

    local float tmp_shared[(NUM_SGS - 1 > 0) ? NUM_SGS - 1 : 1][NCOLS_DST][ROWS_PER_WG][SG_SIZE];

    if (LID_1 > 0) {
        unroll_for (int j = 0; j < NCOLS_DST; j++) {
            unroll_for (int i = 0; i < ROWS_PER_WG; i++) {
                tmp_shared[LID_1 - 1][j][i][LID_0] = tmp[j][i];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (LID_1 > 0) {
        return;
    }

    dst += sample_dst * stride_sample_dst + channel_dst * stride_channel_dst + row0;

    if (IS_MULTI_TOKEN_ID) {
        dst += token_idx * stride_col_dst;
    }

    // sum up partial sums and write back result
    unroll_for (int j = 0; j < NCOLS_DST; j++) {
        unroll_for (int i = 0; i < ROWS_PER_WG; i++) {
            unroll_for (int l = 0; l < NUM_SGS - 1; l++) {
                tmp[j][i] += tmp_shared[l][j][i][LID_0];
            }
            tmp[j][i] = sub_group_reduce_add(tmp[j][i]);
        }

        if (LID_0 < ROWS_PER_WG && 
                (ROWS_PER_WG == 1 || (uint)(row0 + LID_0) < stride_col_dst)) {
            float result = tmp[j][LID_0];
            dst[j * stride_col_dst + LID_0] = result;
        }
    }
}

)";

} // namespace

const char *MulMatQuantVecKernelCode() {
    return g_kernelCodeMulMat;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

