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

const char g_codeMmaF32[] = R"(
inline float4 load_a(const local float *a, const uint stride) {
    const uint tid = get_sub_group_local_id();
    const uint r = tid / 8;
    const uint c = tid % 8;
    float4 t;
    unroll_for (uint i = 0; i < 4; i++) {
        t[i] = a[(r + i * 2) * stride + c];
    }
    return t;
}

inline float8 load_b(const local float *b, const uint stride) {
    const uint tid = get_sub_group_local_id();
    float8 t;
    unroll_for (uint i = 0; i < 8; i++) {
        t[i] = b[tid * stride + i];
    }
    return t;
}

inline void store_c(local float *c, const uint stride, const float8 t) {
    const uint tid = get_sub_group_local_id();
    for (uint i = 0; i < 8; i++) {
        c[tid * stride + i] = t[i];
    }
}

#if MMA_TF32 == 1
inline float8 mma(const float4 a, const float8 b, const float8 acc) {
    return intel_sub_group_tf32_tf32_matrix_mad_k8(a, b, acc);
}

#else
inline float8 mma(const float4 a, const float8 b, const float8 acc) {
    float8 out = acc;
    unroll_for (uint m = 0; m < 8; m++) {
        uint r = m / 2;
        uint c = (m % 2) * 8;
        unroll_for (uint k = 0; k < 8; k++) {
            float ak = sub_group_shuffle(a[r], c + k);
            out[m] = fma(ak, b[k], out[m]);
        }
    }
    return out;
}

#endif

#define TILE_A float4
#define TILE_B float8
#define TILE_C float8

#define LOAD_A(t, a, stride) t = load_a(a, stride)
#define LOAD_B(t, b, stride) t = load_b(b, stride)
#define STORE_C(c, stride, t) store_c(c, stride, t)
#define MMA(acc, a, b) acc = mma(a, b, acc)

)";

const char g_kernelCodeMulMat[] = R"(
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
kernel void mul_mat_mm(
        const global T *x, 
        const global float *y, 
        const global int *ids, 
        global float *dst,
        const int ncols, 
        const int ncols_dst_total, 
        const int nchannels_dst, 
        const int stride_row, 
        const int stride_col_y, 
        const int stride_col_dst,
        const int stride_col_id, 
        const int stride_row_id,
        const int channel_ratio, 
        const int stride_channel_x, 
        const int stride_channel_y, 
        const int stride_channel_dst,
        const int sample_ratio, 
        const int stride_sample_x, 
        const int stride_sample_y, 
        const int stride_sample_dst
        SHAPE_INFO_ARGS) {

    x += X_BASE;
    y += Y_BASE;
    if (ids != NULL) {
        ids += IDS_BASE;
    }
    dst += DST_BASE;

    const int row0 = GID_0 * ROWS_PER_BLOCK;

    int expert_idx = 0;
    int col_base = 0;

    const int channel_dst = HAS_IDS ? 0 : GID_1;

    if (HAS_IDS) {
        // experts + tiles of ncols_dst are packed in the y dimension
        int col_tiles = (ncols_dst_total + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
        const int nchannels_x = GDIM_1 / col_tiles;
        const int tile_idx = GID_1 / nchannels_x;
        expert_idx = GID_1 - tile_idx * nchannels_x;
        col_base = tile_idx * COLS_PER_BLOCK;
    }

    const int channel_x = HAS_IDS ? expert_idx : channel_dst / channel_ratio;
    const int channel_y = channel_dst;
    const int sample_dst = GID_2;
    const int sample_x = sample_dst / sample_ratio;
    const int sample_y = sample_dst;

    x += (long)sample_x * stride_sample_x + channel_x * stride_channel_x  + row0 * stride_row ;
    y += (long)sample_y * stride_sample_y + (HAS_IDS ? 0 : channel_y * stride_channel_y);
    dst += (long)sample_dst * stride_sample_dst + (HAS_IDS ? 0 : channel_dst * stride_channel_dst);

    if (HAS_IDS) {
        const long col_offset = col_base;
        y += col_offset * stride_col_y * Y_STRIDE_SCALE;
        dst += col_offset * stride_col_dst;
        ids += col_offset * stride_row_id;
    }

    const global float2 *y2 = (const global float2 *)y;

    local char shmem[NB_LOCAL];
    local int *slot_map = (local int *)shmem;
    local char *compute_base = HAS_IDS ? shmem + NB_SLOT_MAP : shmem;

    TILE_C C[NTA][NTB] = {0};

    local T *tile_xy = (local T *)compute_base + LID_1 * (TILE_XY_COLS * TILE_K_PADDED);

    if (HAS_IDS) {
        int found = 0;

        for (int j0 = 0; j0 < COLS_PER_BLOCK; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            if (LID_0 == 0) {
                slot_map[j] = -1;
            }

            if (col_base + j >= ncols_dst_total) {
                continue;
            }

            const global int *id_row = ids + j * stride_row_id;

            for (int k = LID_0; k < nchannels_dst; k += SG_SIZE) {
                int match = (id_row[k * stride_col_id] == expert_idx);

                if (match) {
                    slot_map[j] = k;
                    found = 1;
                    break;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (!work_group_any(found)) {
            return;
        }
    }

    for (int col = LID_1 * SG_SIZE + LID_0; col < ncols; col += NUM_SGS * SG_SIZE) {
        TILE_A A[NTA][SG_SIZE / TILE_A_J];
        unroll_for (int itA = 0; itA < NTA; itA++) {
            unroll_for (int i = 0; i < TILE_A_I; i++) {
                tile_xy[i * TILE_K_PADDED + LID_0] = x[(itA * TILE_A_I + i) * stride_row  + col];
            }
            unroll_for (int k0 = 0; k0 < SG_SIZE; k0 += TILE_A_J) {
                LOAD_A(A[itA][k0 / TILE_A_J], tile_xy + k0, TILE_K_PADDED);
            }
        }

        unroll_for (int itB = 0; itB < NTB; itB++) {
#if defined(T_IS_FLOAT)
            unroll_for (int j0 = 0; j0 < TILE_B_I; j0++) {
                const int j = j0 + itB * TILE_B_I;
                if (!HAS_IDS) {
                    tile_xy[j0 * TILE_K_PADDED + LID_0] = 
                        (j < COLS_PER_BLOCK) ? y[j * stride_col_y + col] : 0.0f;
                } else {
                    const bool valid = 
                         (j < COLS_PER_BLOCK && 
                            (col_base + j) < ncols_dst_total && 
                            slot_map[j] >= 0);
                    tile_xy[j0 * TILE_K_PADDED + LID_0] = 
                        valid ? y[slot_map[j] * stride_channel_y + j * stride_col_y + col] : 0.0f;
                }
            }

#else
            unroll_for (int j0 = 0; j0 < TILE_B_I; j0++) {
                const int j = j0 + itB * TILE_B_I;
                if (!HAS_IDS) {
                    const float2 tmp = 
                        (j < COLS_PER_BLOCK) ? y2[j * stride_col_y + col] : (float2)(0.0f, 0.0f);
                    tile_xy[j0 * TILE_K_PADDED + LID_0] = CONVERT_T(tmp);
                } else {
                    const bool valid = 
                        (j < COLS_PER_BLOCK && 
                            (col_base + j) < ncols_dst_total && 
                            slot_map[j] >= 0);
                    float2 tmp = 
                        valid ? 
                            *(const global float2 *)
                                &y[slot_map[j] * stride_channel_y + 2 * (j * stride_col_y + col)] : 
                            (float2)(0.0f, 0.0f);
                    tile_xy[j0 * TILE_K_PADDED + LID_0] = CONVERT_T(tmp);
                }
            }

#endif // defined(T_IS_*)

            unroll_for (int k0 = 0; k0 < SG_SIZE; k0 += TILE_B_J) {
                TILE_B B;
                LOAD_B(B, tile_xy + k0, TILE_K_PADDED);
                unroll_for (int itA = 0; itA < NTA; itA++) {
                    MMA(C[itA][itB], A[itA][k0 / TILE_B_J], B);
                }
            }
        }
    }

    local float *buf_iw = (local float *)compute_base;

    if (NUM_SGS > 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unroll_for (int itB = 0; itB < NTB; itB++) {
        unroll_for (int itA = 0; itA < NTA; itA++) {
            const int i = LID_1 * ROWS_PER_BLOCK + itA * TILE_C_I;
            const int j = itB * TILE_C_J;
            STORE_C(&buf_iw[j * KIW + i], KIW, C[itA][itB]);
        }
    }

    if (NUM_SGS > 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unroll_for (int j0 = 0; j0 < COLS_PER_BLOCK; j0 += NUM_SGS) {
        const int j = j0 + LID_1;

        if (j0 + NUM_SGS > COLS_PER_BLOCK && j >= COLS_PER_BLOCK) {
            return;
        }

        // NE_SUM = ROWS_PER_BLOCK / SG_SIZE
        float sum[NE_SUM] = {0.0f};
        unroll_for (int i0 = 0; i0 < NUM_SGS * ROWS_PER_BLOCK; i0 += ROWS_PER_BLOCK) {
            unroll_for (int i1 = 0; i1 < NE_SUM; i1++) {
                const int i = i0 + i1 * SG_SIZE + LID_0;
                sum[i1] += buf_iw[j * KIW + i];
            }
        }

        if (!HAS_IDS) {
            unroll_for (int i0 = 0; i0 < NE_SUM; i0++) {
                dst[j * stride_col_dst + row0 + i0 * SG_SIZE + LID_0] = sum[i0];
            }
        } else {
            const int slot = (j < COLS_PER_BLOCK) ? slot_map[j] : -1;
            if (slot >= 0 && col_base + j < ncols_dst_total) {
                unroll_for (int i0 = 0; i0 < NE_SUM; i0++) {
                    dst[
                        slot * stride_channel_dst + 
                        j * stride_col_dst + 
                        row0 + 
                        i0 * SG_SIZE + 
                        LID_0] = sum[i0];
                }
            }
        }
    }
}

)";

const char g_kernelCodeMulMatId[] = R"(
inline uint2 fast_div_mod(uint a, uint b, uint b_fd0, uint b_fd1) {
    uint q = fastdiv(a, b_fd0, b_fd1);
    uint r = a - q * b;
    return (uint2)(q, r);
}

#define FAST_DIV_MOD(a, b) fast_div_mod(a, b, b##_fd0, b##_fd1)

// This kernel is for larger batch sizes of mul_mat_id
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
kernel void mul_mat_mm(
        const global T *x, 
        const global float *y,
        const global int *ids_src_compact, 
        const global int *ids_dst_compact,
        const global int *expert_bounds, 
        global float *dst,
        const int ncols, 
        const int ncols_dst_total, 
        const int nchannels_dst, 
        const int stride_row, 
        const int stride_col_y, 
        const int stride_col_dst,
        const int channel_ratio, 
        const int stride_channel_x, 
        const int stride_channel_y, 
        const int stride_channel_dst,
        const int sample_ratio, 
        const int stride_sample_x, 
        const int stride_sample_y, 
        const int stride_sample_dst,
        const uint sis1, 
        const uint sis1_fd0, 
        const uint sis1_fd1, 
        const uint nch,
        const uint nch_fd0,
        const uint nch_fd1
        SHAPE_INFO_ARGS) {

    x += X_BASE;
    y += Y_BASE;
    ids_src_compact += IDS_A_BASE;
    ids_dst_compact += IDS_C_BASE;
    expert_bounds += EXPERT_BOUNDS_BASE;
    dst += DST_BASE;

    const int row0 = GID_0 * ROWS_PER_BLOCK;

    const int expert_idx = GID_1;
    const int expert_start = expert_bounds[expert_idx];
    const int expert_end = expert_bounds[expert_idx + 1];
    const int ncols_expert = expert_end - expert_start;

    const int tiles_for_expert = (ncols_expert + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    const int tile_idx = GID_2;
    if (tile_idx >= tiles_for_expert) {
        return;
    }

    const int col_base = tile_idx * COLS_PER_BLOCK;

    // UNUSED: channel_ratio

    const int channel_x = expert_idx;
    const int sample_dst = 0;
    const int sample_x = sample_dst / sample_ratio;
    const int sample_y = sample_dst;

    x += (long)sample_x * stride_sample_x + channel_x * stride_channel_x + row0 * stride_row;
    y += (long)sample_y * stride_sample_y;
    dst += (long)sample_dst * stride_sample_dst;

    const global int *ids_src_expert = ids_src_compact + expert_start;
    const global int *ids_dst_expert = ids_dst_compact + expert_start;

    local char shmem[NB_LOCAL];
    local char *compute_base = shmem;

    TILE_C C[NTA][NTB] = {0};

    local T *tile_xy = (local T *)compute_base + LID_1 * (TILE_XY_COLS * TILE_K_PADDED);

    for (int col = LID_1 * SG_SIZE + LID_0; col < ncols; col += NUM_SGS * SG_SIZE) {
        TILE_A A[NTA][SG_SIZE / TILE_A_J];
        unroll_for (int itA = 0; itA < NTA; itA++) {
            unroll_for (int i = 0; i < TILE_A_I; i++) {
                tile_xy[i * TILE_K_PADDED + LID_0] = x[(itA * TILE_A_I + i) * stride_row  + col];
            }
            unroll_for (int k0 = 0; k0 < SG_SIZE; k0 += TILE_A_J) {
                LOAD_A(A[itA][k0 / TILE_A_J], tile_xy + k0, TILE_K_PADDED);
            }
        }

#if defined(T_IS_FLOAT)
        float vals_buf[2][TILE_B_I];

#define GATHER_TILE(_tile_idx_local, _vals) { \
            const int tile_idx_local = _tile_idx_local; \
            float *vals = _vals; \
            unroll_for (int j0 = 0; j0 < TILE_B_I; j0++) { \
                const int j = j0 + tile_idx_local * TILE_B_I; \
                const int global_j = col_base + j; \
                float val = 0.0f; \
                if (j < COLS_PER_BLOCK && global_j < ncols_expert) { \
                    const int src_entry = ids_src_expert[global_j]; \
                    const uint2 qrm = FAST_DIV_MOD((uint)src_entry, sis1); \
                    const int token = (int)qrm.x; \
                    const int channel = (int)qrm.y; \
                    if (token < ncols_dst_total) { \
                        val = y[channel * stride_channel_y + token * stride_col_y + col]; \
                    } \
                } \
                vals[j0] = val; \
            } \
        }

        GATHER_TILE(0, vals_buf[0]);

        int curr_buf = 0;
        int next_buf = 1;
        unroll_for (int itB = 0; itB < NTB; itB++) {
            unroll_for (int j0 = 0; j0 < TILE_B_I; j0++) {
                tile_xy[j0 * TILE_K_PADDED + LID_0] = vals_buf[curr_buf][j0];
            }

            if (itB + 1 < NTB) {
                GATHER_TILE(itB + 1, vals_buf[next_buf]);
            }

            unroll_for (int k0 = 0; k0 < SG_SIZE; k0 += TILE_B_J) {
                TILE_B B;
                LOAD_B(B, tile_xy + k0, TILE_K_PADDED);
                unroll_for (int itA = 0; itA < NTA; itA++) {
                    MMA(C[itA][itB], A[itA][k0 / TILE_B_J], B);
                }
            }

            if (itB + 1 < NTB) {
                curr_buf ^= 1;
                next_buf ^= 1;
            }
        }
#undef GATHER_TILE

#else
        float2 vals_buf[2][TILE_B_I];

#define GATHER_TILE(_tile_idx_local, _vals) { \
            const int tile_idx_local = _tile_idx_local; \
            float2 *vals = _vals; \
            unroll_for (int j0 = 0; j0 < TILE_B_I; j0++) { \
                const int j = j0 + tile_idx_local * TILE_B_I; \
                const int global_j = col_base + j; \
                float2 tmp = (float2)(0.0f, 0.0f); \
                if (j < COLS_PER_BLOCK && global_j < ncols_expert) { \
                    const int src_entry = ids_src_expert[global_j]; \
                    const uint2 qrm = FAST_DIV_MOD((uint)src_entry, sis1); \
                    const int token = (int)qrm.x; \
                    const int channel = (int)qrm.y; \
                    if (token < ncols_dst_total) { \
                        tmp = \
                            *(const global float2 *) \
                                &y[channel * stride_channel_y + 2 * (token * stride_col_y + col)]; \
                    } \
                } \
                vals[j0] = tmp; \
            } \
        }

        // ACHTUNG: Is not this always true?
        if (NTB > 0) {
            GATHER_TILE(0, vals_buf[0]);
        }

        int curr_buf = 0;
        int next_buf = 1;
        unroll_for (int itB = 0; itB < NTB; itB++) {
            unroll_for (int j0 = 0; j0 < TILE_B_I; j0++) {
                const float2 tmp = vals_buf[curr_buf][j0];
                tile_xy[j0 * TILE_K_PADDED + LID_0] = CONVERT_T(tmp);
            }

            if (itB + 1 < NTB) {
                GATHER_TILE(itB + 1, vals_buf[next_buf]);
            }

            unroll_for (int k0 = 0; k0 < SG_SIZE; k0 += TILE_B_J) {
                TILE_B B;
                LOAD_B(B, tile_xy + k0, TILE_K_PADDED);
                unroll_for (int itA = 0; itA < NTA; itA++) {
                    MMA(C[itA][itB], A[itA][k0 / TILE_B_J], B);
                }
            }

            if (itB + 1 < NTB) {
                curr_buf ^= 1;
                next_buf ^= 1;
            }
        }
#undef GATHER_TILE

#endif // defined(T_IS_*)
    }

    local float *buf_iw = (local float *)compute_base;

    if (NUM_SGS > 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unroll_for (int itB = 0; itB < NTB; itB++) {
        unroll_for (int itA = 0; itA < NTA; itA++) {
            const int i = LID_1 * ROWS_PER_BLOCK + itA * TILE_C_I;
            const int j = itB * TILE_C_J;
            STORE_C(&buf_iw[j * KIW + i], KIW, C[itA][itB]);
        }
    }

    if (NUM_SGS > 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unroll_for (int j0 = 0; j0 < COLS_PER_BLOCK; j0 += NUM_SGS) {
        const int j = j0 + LID_1;

        if (j0 + NUM_SGS > COLS_PER_BLOCK && j >= COLS_PER_BLOCK) {
            return;
        }

        // NE_SUM = ROWS_PER_BLOCK / SG_SIZE
        float sum[NE_SUM] = {0.0f};
        unroll_for (int i0 = 0; i0 < NUM_SGS * ROWS_PER_BLOCK; i0 += ROWS_PER_BLOCK) {
            unroll_for (int i1 = 0; i1 < NE_SUM; i1++) {
                const int i = i0 + i1 * SG_SIZE + LID_0;
                sum[i1] += buf_iw[j * KIW + i];
            }
        }

        const int global_j = col_base + j;
        if (j < COLS_PER_BLOCK && global_j < ncols_expert && nchannels_dst > 0) {
            const int dst_entry = ids_dst_expert[global_j];
            const uint2 qrm = FAST_DIV_MOD((uint)dst_entry, nch);
            const int token = (int)qrm.x;
            if (token < ncols_dst_total) {
                const int slot = (int)qrm.y;
                unroll_for (int i0 = 0; i0 < NE_SUM; i0++) {
                    dst[
                        slot * stride_channel_dst + 
                        token * stride_col_dst + 
                        row0 + 
                        i0 * SG_SIZE + 
                        LID_0] = sum[i0];
                }
            }
        }
    }
} 

)";

} // namespace

const char *MulMatMmaF32Code() {
    return g_codeMmaF32;
}

const char *MulMatMmKernelCode() {
    return g_kernelCodeMulMat;
}

const char *MulMatIdMmKernelCode() {
    return g_kernelCodeMulMatId;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

