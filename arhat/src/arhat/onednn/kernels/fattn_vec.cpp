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

const char g_kernelCodeFattn[] = R"(
#define MAD_FH2(acc, u, v) { \
    acc += u.x * v.x; \
    acc += u.y * v.y; \
} 

#define ALIGNED __attribute__ ((aligned(16)))

#define unroll_for __attribute__((opencl_unroll_hint)) for

#define GDIM_0 get_num_groups(0)
#define GDIM_1 get_num_groups(1)

#define GID_0 get_group_id(0)
#define GID_1 get_group_id(1)
#define GID_2 get_group_id(2)

#define LID_0 get_local_id(0)
#define LID_1 get_local_id(1)

#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f) 

inline float get_alibi_slope(
        const float max_bias, 
        const uint h, 
        const uint n_head_log2, 
        const float m0, 
        const float m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = (h < n_head_log2) ? m0 : m1;
    const int exph = (h < n_head_log2) ? h + 1 : 2 * (h - n_head_log2) + 1;

    return pown(base, exph);
} 

inline float vec_dot_KQ_f16(const global half *K_c, const half2 *Q_v) {
    const global half2 *K_h2 = (const global half2 *)K_c;

    float sum = 0.0f;

    unroll_for (int k_KQ_0 = 0; k_KQ_0 < D / 2; k_KQ_0 += NTHREADS_KQ * CPY_NE) {
        half2 tmp[CPY_NE] ALIGNED;
        COPY_RG_H2(CPY_NE, tmp, K_h2 + k_KQ_0 + (LID_0 % NTHREADS_KQ) * CPY_NE);
        unroll_for (int k_KQ_1 = 0; k_KQ_1 < CPY_NE; k_KQ_1++) {
            MAD_FH2(sum, tmp[k_KQ_1], ((const half2 *)Q_v)[k_KQ_0 / NTHREADS_KQ + k_KQ_1]);
        }
    }

    return sum;
} 

__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void fattn_vec(
        const global float *Q,
        const global half *K,
        const global half *V,
        const global half *mask,
        const global float *sinks,
        const global int *KV_max,
        global float *dst,
        global float2 *dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const float logit_softcap
        SHAPE_INFO_ARGS) {

    // Index of the Q / QKV column to work on
    const int ic0 = GID_0 * NCOLS; 

    const int sequence = GID_2 / Q_D1;
    const int head = GID_2 - sequence * Q_D1;
    Q += Q_BASE + Q_S0 * sequence + Q_S1 * head + Q_S2 * ic0;
    // With grouped query attention there are > 1 Q matrices per K, V matrix
    K += K_BASE + K_S0 * sequence + K_S1 * (head / GQA_RATIO);
    V += V_BASE + V_S0 * sequence + V_S1 * (head / GQA_RATIO);

    global const half *maskh = mask + MASK_BASE + MASK_S0 * (sequence % MASK_D0) + MASK_S2 * ic0;

    const float slope = get_alibi_slope(max_bias, head, N_HEAD_LOG2, m0, m1);

    const int tid = SG_SIZE * LID_1 + LID_0;

    half2 VKQ[NCOLS][(D / 2) / NTHREADS_V] = {{{0.0f, 0.0f}}};
    local half KQ[(NE_KQ > NE_COMBINE) ? NE_KQ : NE_COMBINE];

    float KQ_max[NCOLS];
    float KQ_sum[NCOLS];
    unroll_for (int j = 0; j < NCOLS; j++) {
        KQ_max[j] = -FLT_MAX / 2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 and store in registers

    // Will be initialized completely
    half2 Q_reg[NCOLS][(D / 2) / NTHREADS_KQ]; 

    const half2 scale_h2 = (half2)(scale, scale);
    unroll_for (int j = 0; j < NCOLS; j++) {
        global const float2 *Q_j = (global const float2 *)(Q + j * Q_S2);
        unroll_for (int i0 = 0; i0 < D / 2; i0 += NTHREADS_KQ * CPY_NE) {
            const int i = i0 + ((NTHREADS_KQ == SG_SIZE) ? LID_0 : LID_0 % NTHREADS_KQ) * CPY_NE;
            float2 tmp[CPY_NE] ALIGNED = {{0.0f, 0.0f}};
            if (NCOLS == 1 || ic0 + j < Q_D2) {
                COPY_RG_F(CPY_NE, tmp, &Q_j[i]);
                COPY_RG_F(CPY_NE, tmp + CPY_NE / 2, &Q_j[i + CPY_NE / 2]);
            }
            unroll_for (int i1 = 0; i1 < CPY_NE; i1++) {
                Q_reg[j][i0 / NTHREADS_KQ + i1] = (half2)(tmp[i1].x, tmp[i1].y);
            }
        }
        unroll_for (int k = 0; k < (D / 2) / NTHREADS_KQ; k++) {
            Q_reg[j][k] *= scale_h2;
        }
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence * GDIM_0 + GID_0] : K_D2;
    K += GID_1 * NTHREADS * K_S2;
    V += GID_1 * NTHREADS * V_S2;
    maskh += GID_1 * NTHREADS;
    for (int k_VKQ_0 = GID_1 * NTHREADS; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += GDIM_1 * NTHREADS) {
        // Calculate KQ tile and keep track of new maximum KQ values
        // KQ in registers
        float KQ_reg[NCOLS]; 

        float KQ_max_new[NCOLS];
        unroll_for (int j = 0; j < NCOLS; j++) {
            KQ_max_new[j] = KQ_max[j];
        }

        unroll_for (int i_KQ_0 = 0; i_KQ_0 < NTHREADS_KQ; i_KQ_0++) {
            const int i_KQ = 
                LID_1 * SG_SIZE + 
                ((NTHREADS_KQ == SG_SIZE) ? 0 : (LID_0 & ~(NTHREADS_KQ - 1))) + 
                i_KQ_0;

            unroll_for (int j = 0; j < NCOLS; j++) {
                float sum = vec_dot_KQ_f16(K + i_KQ * K_S2, Q_reg[j]);
                // requires cl_khr_subgroup_clustered_reduce extension
                sum = sub_group_clustered_reduce_add(sum, NTHREADS_KQ);

                if (USE_LOGIT_SOFTCAP) {
                    sum = logit_softcap * tanh(sum);
                }

                if (mask && (NCOLS == 1 || ic0 + j < Q_D2)) {
                    sum += slope * convert_float(maskh[j * K_D2 + i_KQ]);
                }

                KQ_max_new[j] = fmax(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if (((NTHREADS_KQ == SG_SIZE) ? LID_0 : LID_0 % NTHREADS_KQ) == (uint)i_KQ_0) {
                    KQ_reg[j] = sum;
                }
            }
        }

        unroll_for (int j = 0; j < NCOLS; j++) {
            unroll_for (int offset = NTHREADS_KQ; offset < SG_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmax(KQ_max_new[j], intel_sub_group_shuffle_xor(KQ_max_new[j], offset));
            }
            const float KQ_max_scale = exp(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = exp(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j] * KQ_max_scale + KQ_reg[j];
            KQ[j * NTHREADS + tid] = KQ_reg[j];

            const half2 KQ_max_scale_h2 = (half2)(KQ_max_scale, KQ_max_scale);
            unroll_for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2; i_VKQ_0 += NTHREADS_V) {
                VKQ[j][i_VKQ_0 / NTHREADS_V] *= KQ_max_scale_h2;
            }
        }

        // ensure correct oredring of operations on local KQ[]
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        unroll_for (int k0 = 0; k0 < SG_SIZE; k0 += V_COLS_PER_ITER) {
            const int k = LID_1 * SG_SIZE + k0 + ((NTHREADS_V == SG_SIZE) ? 0 : LID_0 / NTHREADS_V);

            half2 KQ_k[NCOLS];
            unroll_for (int j = 0; j < NCOLS; j++) {
                KQ_k[j] = (half2)(KQ[j * NTHREADS + k], KQ[j * NTHREADS + k]);
            }
            unroll_for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2; i_VKQ_0 += NTHREADS_V * (V_ROWS_PER_THREAD / 2)) {
                half2 tmp[V_ROWS_PER_THREAD / 2];
                // "dequantize"
                COPY_RG_H(
                    V_ROWS_PER_THREAD,
                    tmp,
                    &V[
                        k * V_S2 +
                        2 * i_VKQ_0 + 
                        ((NTHREADS_V == SG_SIZE) ? LID_0 : LID_0 % NTHREADS_V) * V_ROWS_PER_THREAD]);
                unroll_for (int i_VKQ_1 = 0; i_VKQ_1 < V_ROWS_PER_THREAD / 2; i_VKQ_1++) {
                    unroll_for (int j = 0; j < NCOLS; j++) {
                        VKQ[j][i_VKQ_0 / NTHREADS_V + i_VKQ_1] += tmp[i_VKQ_1] * KQ_k[j];
                    }
                }
            }
        }

        // Increment pointers after each loop
        K += GDIM_1 * NTHREADS * K_S2;
        V += GDIM_1 * NTHREADS * V_S2; 
        maskh += GDIM_1 * NTHREADS;
    }

    if (sinks && GID_1 == 0) {
        const float sink = sinks[head];

        unroll_for (int j0 = 0; j0 < NCOLS; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            if (j0 + NUM_SGS > NCOLS && j >= NCOLS) {
                break;
            }

            const float kqmax_new_j = fmax(sink, KQ_max[j]);
            const float KQ_max_scale = exp(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale + ((LID_0 == 0) ? exp(sink - KQ_max[j]) : 0.0f);

            const half2 KQ_max_scale_h2 = (half2)(KQ_max_scale, KQ_max_scale);
            unroll_for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2; i_VKQ_0 += NTHREADS_V) {
                VKQ[j][i_VKQ_0 / NTHREADS_V] *= KQ_max_scale_h2;
            }
        }
    }

    local float KQ_max_shared[NCOLS][SG_SIZE];
    local float KQ_sum_shared[NCOLS][SG_SIZE];
    unroll_for (int j = 0; j < NCOLS; j++) {
        if (LID_1 == 0) {
            KQ_max_shared[j][LID_0] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j][LID_0] = 0.0f;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unroll_for (int j = 0; j < NCOLS; j++) {
        if (LID_0 == 0) {
            KQ_max_shared[j][LID_1] = KQ_max[j];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unroll_for (int j_VKQ = 0; j_VKQ < NCOLS; j_VKQ++) {
        if (NCOLS > 1 && ic0 + j_VKQ >= Q_D2) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][LID_0];
        kqmax_new = sub_group_reduce_max(kqmax_new);
        const float kqmax_scale = exp(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

        local half2 *VKQ_tmp = 
            (local half2 *)KQ + 
            LID_1 * (V_COLS_PER_ITER * D / 2) + 
            ((NTHREADS_V == SG_SIZE) ? 0 : LID_0 / NTHREADS_V) * (D / 2);

        const half2 kqmax_scale_h2 = (half2)(kqmax_scale, kqmax_scale);
        unroll_for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2; i_VKQ_0 += NTHREADS_V) {
            VKQ[j_VKQ][i_VKQ_0 / NTHREADS_V] *= kqmax_scale_h2;
        }
        unroll_for (int i_VKQ_0 = 0; i_VKQ_0 < D / 2; i_VKQ_0 += NTHREADS_V * (V_ROWS_PER_THREAD / 2)) {
            const int i_VKQ = 
                i_VKQ_0 + 
                ((NTHREADS_V == SG_SIZE) ? LID_0 : LID_0 % NTHREADS_V) * (V_ROWS_PER_THREAD / 2);

            COPY_LR_H(
                V_ROWS_PER_THREAD,
                VKQ_tmp + i_VKQ, 
                &VKQ[j_VKQ][i_VKQ_0 / NTHREADS_V]);
        }

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = sub_group_reduce_add(KQ_sum[j_VKQ]);
        if (LID_0 == 0) {
            KQ_sum_shared[j_VKQ][LID_1] = KQ_sum[j_VKQ];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (NTHREADS <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][LID_0];
            KQ_sum[j_VKQ] = sub_group_reduce_add(KQ_sum[j_VKQ]);

            unroll_for (int i0 = 0; i0 < D; i0 += NTHREADS) {
                float dst_val = 0.0f;
                unroll_for (int w = 0; w < NUM_SGS; w++) {
                    unroll_for (int v = 0; v < V_COLS_PER_ITER; v++) {
                        dst_val += convert_float(KQ[w * V_COLS_PER_ITER * D + v * D + i0 + tid]);
                    }
                }
                if (GDIM_1 == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * Q_D2 + ic0 + j_VKQ) * Q_D1 + head) * GDIM_1 + GID_1) * D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < NCOLS - 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }

    }

    if (GDIM_1 != 1 && tid < NCOLS && (NCOLS == 1 || ic0 + tid < Q_D2)) {
        dst_meta[((sequence * Q_D2 + ic0 + tid) * Q_D1 + head) * GDIM_1 + GID_1] = 
            (float2)(KQ_max[tid], KQ_sum[tid]);
    }
} 

)";

} // namespace

const char *FattnVecKernelCode() {
    return g_kernelCodeFattn;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

