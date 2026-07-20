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
    float2 tu = convert_float2(u); \
    float2 tv = convert_float2(v); \
    acc += tu.x * tv.x; \
    acc += tu.y * tv.y; \
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

inline void fattn_tile_load_tile(
        const int I,
        const int J,
        const int J_padding,
        const int oob_check,
        const global half2 *KV, 
        local half2 *tile_KV, 
        const int stride_KV, 
        const int i_sup) {
    // 1: max 64 * 16 = 512 bytes, 512 half
    // 2: max 32 * 16 = 512 bytes, 256 half
    // 3: max 16 * 16 = 256 bytes, 128 half
    // 4: max  8 * 16 = 128 bytes,  64 half
    // 5: max  4 * 16 =  64 bytes,  32 half
    // 6: max  2 * 16 =  32 bytes,  16 half
    // 7: max  1 * 16 =  16 bytes,   8 half
    unroll_for (int n = 0; n < 7; n++) {
        const int stride_j = SG_SIZE >> n;

        if (stride_j != 0) {
            const int j0_start = 
                (stride_j == SG_SIZE) ? 
                    0 : 
                    ((J / 2) / CPY_NE) - ((J / 2) / CPY_NE) % (2 * stride_j);
            const int j0_stop  = ((J / 2) / CPY_NE) - ((J / 2) / CPY_NE) % (1 * stride_j);
            const int stride_i = SG_SIZE / stride_j;

            if (j0_start != j0_stop) {
                unroll_for (int i0 = 0; i0 < I; i0 += NUM_SGS * stride_i) {
                    const int i = 
                        i0 + 
                        LID_1 * stride_i + 
                        ((stride_j == SG_SIZE) ? 0 : LID_0 / stride_j);

                    if (i0 + NUM_SGS * stride_i <= I || i < I) {
                        unroll_for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                            const int j = 
                                j0 * CPY_NE + 
                                ((stride_j == SG_SIZE) ? LID_0 : LID_0 % stride_j) * CPY_NE;

                            if (!oob_check || i < i_sup) {
                                COPY_LG_H2(
                                    CPY_NE, 
                                    tile_KV + i * (J / 2 + J_padding) + j,
                                    KV + i * stride_KV + j);
                            } else {
                                half2 zero[CPY_NE] ALIGNED = {{0.0f, 0.0f}};
                                COPY_LR_H2(
                                    CPY_NE, 
                                    tile_KV + i * (J / 2 + J_padding) + j,
                                    zero);
                            }
                        }
                    }
                }
            }           
        }
    }
}

// Performs a single iteration in for the KQ matrix multiplication
inline void fattn_tile_iter_KQ(
        const int oob_check,
        local half2 *Q_tmp,
        const global half2 *K_h2,
        local half2 *KV_tmp,
        const int stride_K2,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float *KQ_acc) {
    fattn_tile_load_tile(
        NBATCH_FA,
        NBATCH_K,
        CPY_NE, 
        oob_check,
        K_h2 + (long)k_VKQ_0 * stride_K2 + k_KQ_0/2, 
        KV_tmp, 
        stride_K2, 
        k_VKQ_sup);
    barrier(CLK_LOCAL_MEM_FENCE);

    unroll_for (int k_KQ_1 = 0; k_KQ_1 < NBATCH_K / 2; k_KQ_1 += CPY_NE) {
        half2 K_k[NBATCH_FA / (NP * SG_SIZE)][CPY_NE] ALIGNED;
        half2 Q_k[CPSG][CPY_NE] ALIGNED;

        unroll_for (int i_KQ_0 = 0; i_KQ_0 < NBATCH_FA; i_KQ_0 += NP * SG_SIZE) {
            const int i_KQ = i_KQ_0 + (LID_1 % NP) * SG_SIZE + LID_0;

            COPY_RL_H2(
                CPY_NE, 
                &K_k[i_KQ_0 / (NP * SG_SIZE)], 
                &KV_tmp[i_KQ * (NBATCH_K / 2 + CPY_NE) + k_KQ_1]);
        }
        unroll_for (int jc0 = 0; jc0 < CPSG; jc0++) {
            const int jc = jc0 + (LID_1 / NP) * CPSG;

            COPY_RL_H2(
                CPY_NE,
                &Q_k[jc0], 
                &Q_tmp[jc * (DKQ / 2) + k_KQ_0 / 2 + k_KQ_1]);
        }

        unroll_for (int i_KQ_0 = 0; i_KQ_0 < NBATCH_FA; i_KQ_0 += NP * SG_SIZE) {
            unroll_for (int jc0 = 0; jc0 < CPSG; jc0++) {
                unroll_for (int k = 0; k < CPY_NE; k++) {
                    MAD_FH2(
                        KQ_acc[i_KQ_0 / (NP * SG_SIZE) * CPSG + jc0], 
                        K_k[i_KQ_0 / (NP * SG_SIZE)][k], 
                        Q_k[jc0][k]);
                }
            }
        }
    }

    // Sync not needed on last iteration.
    if (k_KQ_0 + NBATCH_K < DKQ) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }
} 

// Performs a single iteration of the main loop over up to NBATCH_FA tokens
inline void fattn_tile_iter(
        const int oob_check,
        local half2 *Q_tmp,
        const global half2 *K_h2,
        const global half2 *V_h2,
        const global half *mask,
        const int Q_D2, // TODO: Implement fastdiv
        const float logit_softcap,
        const float slope,
        local half *KQ,
        local half2 *KV_tmp,
        const int stride_K2,
        const int stride_V2,
        const int stride_mask,
        float *KQ_max,
        float *KQ_sum,
        half2 *VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        const int col_Q_0,
        local float *KQ_max_new_shared) {
    // k supremum, only smaller k values have valid KV data
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; 

    float KQ_max_new[CPSG];
    unroll_for (int jc0 = 0; jc0 < CPSG; jc0++) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    // Accumulators for KQ matrix multiplication.
    float KQ_acc[NBATCH_FA / (NP * SG_SIZE) * CPSG] = {0.0f}; 

    // KQ = K @ Q matrix multiplication:
    const int nbatch_K_last = DKQ % NBATCH_K;
    unroll_for (int k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += NBATCH_K) {
        fattn_tile_iter_KQ(
            oob_check,
            Q_tmp, 
            K_h2, 
            KV_tmp, 
            stride_K2, 
            k_VKQ_0, 
            k_VKQ_sup, 
            k_KQ_0, 
            KQ_acc);
    }
    if (nbatch_K_last > 0) {
        const int k_KQ_0 = DKQ - nbatch_K_last;
        fattn_tile_iter_KQ(
            oob_check,
            Q_tmp, 
            K_h2, 
            KV_tmp, 
            stride_K2, 
            k_VKQ_0, 
            k_VKQ_sup, 
            k_KQ_0, 
            KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
    unroll_for (int jc0 = 0; jc0 < CPSG; jc0++) {
        const int j = (col_Q_0 + (jc0 + (LID_1 / NP) * CPSG) / NCOLS2) % Q_D2;

        unroll_for (int i_KQ_0 = 0; i_KQ_0 < NBATCH_FA; i_KQ_0 += NP * SG_SIZE) {
            const int i_KQ = i_KQ_0 + (LID_1 % NP) * SG_SIZE + LID_0;

            if (USE_LOGIT_SOFTCAP) {
                KQ_acc[(i_KQ_0 / (NP * SG_SIZE)) * CPSG + jc0] = 
                    logit_softcap * tanh(KQ_acc[(i_KQ_0 / (NP * SG_SIZE)) * CPSG + jc0]);
            }

            if (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0 / (NP * SG_SIZE)) * CPSG + jc0] += 
                    (NCOLS2 > 1 || mask) ?
                        slope * convert_float(mask[j * stride_mask + k_VKQ_0 + i_KQ]) : 
                        0.0f;

                KQ_max_new[jc0] = 
                    fmax(
                        KQ_max_new[jc0], 
                        KQ_acc[(i_KQ_0 / (NP * SG_SIZE)) * CPSG + jc0] + FATTN_KQ_MAX_OFFSET);
            }
        }

        KQ_max_new[jc0] = sub_group_reduce_max(KQ_max_new[jc0]);
    }

    if (NP == 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        if (LID_0 == 0) {
            KQ_max_new_shared[LID_1] = KQ_max_new[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        KQ_max_new[0] = KQ_max_new_shared[(LID_1 & ~(NP - 1)) + LID_0 % NP];
        // requires cl_khr_subgroup_clustered_reduce extension
        KQ_max_new[0] = sub_group_clustered_reduce_max(KQ_max_new[0], NP);
    }

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators
    unroll_for (int jc0 = 0; jc0 < CPSG; jc0 += KQ_CS) {
        half tmp[NBATCH_FA / (NP * SG_SIZE)][KQ_CS] ALIGNED;

        unroll_for (int jc1 = 0; jc1 < KQ_CS; jc1++) {
            const int jc = jc0 + jc1;

            const float KQ_max_scale = exp(KQ_max[jc] - KQ_max_new[jc]);
            KQ_max[jc] = KQ_max_new[jc];

            float KQ_sum_add = 0.0f;
            unroll_for (int i0 = 0; i0 < NBATCH_FA; i0 += NP * SG_SIZE) {
                const float val = 
                    (!oob_check || i0 + (LID_1 % NP) * SG_SIZE + LID_0 < (uint)k_VKQ_sup) ?
                        exp(KQ_acc[(i0 / (NP * SG_SIZE)) * CPSG + jc] - KQ_max[jc]) : 
                        0.0f;
                KQ_sum_add += val;
                tmp[i0 / (NP * SG_SIZE)][jc1] = val;
            }
            KQ_sum[jc] = KQ_sum[jc] * KQ_max_scale + KQ_sum_add;

            const half2 KQ_max_scale_h2 = (half2)(KQ_max_scale, KQ_max_scale);
            unroll_for (int i0 = 0; i0 < DV_P / 2; i0 += SG_SIZE) {
                VKQ[jc * ((DV_P / 2) / SG_SIZE) + i0 / SG_SIZE] *= KQ_max_scale_h2;
            }
        }

        unroll_for (int i0 = 0; i0 < NBATCH_FA; i0 += NP * SG_SIZE) {
            const int i = i0 + (LID_1 % NP) * SG_SIZE + LID_0;

            COPY_LR_H(
                KQ_CS,
                &KQ[ 
                    (jc0 / KQ_CS + (LID_1 / NP) * (CPSG / KQ_CS)) * (NBATCH_FA * KQ_CS) + 
                    i * KQ_CS],
                tmp[i0 / (NP * SG_SIZE)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    unroll_for (int k0 = 0; k0 < NBATCH_FA; k0 += NBATCH_V) {
        fattn_tile_load_tile(
            NBATCH_V,
            DV,
            0,
            oob_check,
            V_h2 + (long)(k_VKQ_0 + k0) * stride_V2, 
            KV_tmp, 
            stride_V2, 
            k_VKQ_sup - k0);
        barrier(CLK_LOCAL_MEM_FENCE);

        unroll_for (int k1 = 0; k1 < NBATCH_V; k1 += NP) {
            half2 V_k[(DV_P / 2) / SG_SIZE] ALIGNED;
            half2 KQ_k[CPSG] ALIGNED;

            unroll_for (int i0 = 0; i0 < DV_P / 2; i0 += SG_SIZE * CPY_NE_DV2) {
                COPY_RL_H2(
                    CPY_NE_DV2,
                    &V_k[i0 / SG_SIZE], 
                    &KV_tmp[
                        (k1 + LID_1 % NP) * (DV / 2) + 
                        i0 + 
                        LID_0 * CPY_NE_DV2]);
            }
            unroll_for (int jc_VKQ_0 = 0; jc_VKQ_0 < CPSG; jc_VKQ_0 += KQ_CS) {
                const int jc_KQ = jc_VKQ_0 / KQ_CS + (LID_1 / NP) * (CPSG / KQ_CS);

                half tmp[KQ_CS] ALIGNED;
                COPY_RL_H(
                    KQ_CS,
                    &tmp, 
                    &KQ[
                        jc_KQ * (NBATCH_FA * KQ_CS) + 
                        (k0 + k1 + LID_1 % NP) * KQ_CS]);
                unroll_for (int jc_VKQ_1 = 0; jc_VKQ_1 < KQ_CS; jc_VKQ_1++) {
                    KQ_k[jc_VKQ_0 + jc_VKQ_1] = (half2)(tmp[jc_VKQ_1], tmp[jc_VKQ_1]);
                }
            }

            unroll_for (int i0 = 0; i0 < DV_P / 2; i0 += SG_SIZE) {
                unroll_for (int jc_VKQ_0 = 0; jc_VKQ_0 < CPSG; jc_VKQ_0++) {
                    VKQ[jc_VKQ_0 * ((DV_P / 2) / SG_SIZE) + i0 / SG_SIZE] += 
                        V_k[i0 / SG_SIZE] * KQ_k[jc_VKQ_0];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
)"
// Split to circumvent MSVC restrictions on string literal size
R"(
__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void fattn_tile(
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
    // Index of the first Q column for this work-group to work on
    const int col_Q_0 = GID_0 * NCOLS1;

    const int sequence = GID_2 / (Q_D1 / NCOLS2);
    // GID_2 % (Q_D1 / NCOLS2)
    const int head0 = GID_2 * NCOLS2 - sequence * Q_D1; 
    // With grouped query attention there are > 1 Q matrices per K, V matrix.
    global const float *Q_f = 
        (global const float *)(Q + Q_BASE + Q_S0 * sequence + Q_S1 * head0);
    // K and V have same shape
    global const half2 *K_h2 = 
        (global const half2 *)(K + K_BASE + K_S0 * sequence + K_S1 * (head0 / GQA_RATIO));
    global const half2 *V_h2 = 
        (global const half2 *)(V + V_BASE + V_S0 * sequence + V_S1 * (head0 / GQA_RATIO));

    global const half *maskh = 
        mask ? 
            (global const half *)(mask + MASK_BASE + MASK_S0 * (sequence % MASK_D0)) : 
            NULL;

    // divide by 2 to account for half2
    const int stride_K2 = K_S2 / 2;
    const int stride_V2 = V_S2 / 2;
    const int stride_mask = MASK_S2; 

    const float slope = 
        (NCOLS2 == 1) ? 
            get_alibi_slope(max_bias, head0, N_HEAD_LOG2, m0, m1) : 
            1.0f;

    // Q_tmp     local buffer to hold Q data for the entire lifetime of the kernel
    // KV_tmp    local buffer to hold fragments of K/V data while iterating over K_D2
    //           padded to avoid memory conflicts for K (CPY_NE) and OOB accesses for V (DV_P - DV)
    // KQ        local buffer to hold KQ fragments between KQ and VKQ matrix multiplications
    // VKQ       Accumulators in registers for the final VKQ result

    local half2 Q_tmp[NCOLS * DKQ / 2];
    local half2 KV_tmp[NBATCH_FA * (NBATCH_K / 2 + CPY_NE) + DV_P - DV];
    local half KQ[NCOLS * NBATCH_FA];
    half2 VKQ[CPSG * ((DV_P / 2) / SG_SIZE)] ALIGNED = {{0.0f, 0.0f}}; 

    float KQ_max[CPSG];
    unroll_for (int j0 = 0; j0 < NCOLS; j0 += NUM_SGS) {
        KQ_max[j0 / NUM_SGS] = -FLT_MAX / 2.0f;
    }
    float KQ_sum[CPSG] = {0.0f};
 
    // Load Q data, convert to FP16 if fast:
    unroll_for (int jc0 = 0; jc0 < CPSG; jc0++) {
        const int jc = jc0 + (LID_1 / NP) * CPSG;

        const int j = jc / NCOLS2;
        const int c = jc % NCOLS2;

        unroll_for (int i0 = 0; i0 < DKQ_P; i0 += NP * SG_SIZE * CPY_NE_DKQ) {
            if (i0 + NP * SG_SIZE * CPY_NE_DKQ <= DKQ || 
                    i0 + (LID_1 % NP) * (SG_SIZE * CPY_NE_DKQ) + LID_0 * CPY_NE_DKQ < DKQ) {
                float tmp_f[CPY_NE_DKQ] ALIGNED = {0.0f};
                COPY_RG_F(
                    CPY_NE_DKQ,
                    tmp_f, 
                    &Q_f[
                        c * Q_S1 + 
                        ((col_Q_0 + j) % Q_D2) * Q_S2 +
                        i0 + 
                        (LID_1 % NP) * (SG_SIZE * CPY_NE_DKQ) + 
                        LID_0 * CPY_NE_DKQ]);

                unroll_for (int i1 = 0; i1 < CPY_NE_DKQ; i1++) {
                    tmp_f[i1] *= scale;
                }

                half2 tmp_h2[CPY_NE_DKQ / 2] ALIGNED;
                unroll_for (int i1 = 0; i1 < CPY_NE_DKQ; i1 += 2) {
                    tmp_h2[i1 / 2] = (half2)(tmp_f[i1 + 0], tmp_f[i1 + 1]);
                }
                COPY_LR_H2(
                    CPY_NE_DKQ2,
                    &Q_tmp[
                        jc * (DKQ / 2) + 
                        i0 / 2 + 
                        (LID_1 % NP) * (SG_SIZE * CPY_NE_DKQ / 2) + 
                        LID_0 * (CPY_NE_DKQ / 2)],
                    tmp_h2);
            }
        }
    } 

    barrier(CLK_LOCAL_MEM_FENCE);

    // Moved from fattn_tile_iter
    local float KQ_max_new_shared[NUM_SGS];

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence * GDIM_0 + GID_0] : K_D2;
    if (NCOLS2 == 1) {
        // Branch with out-of-bounds checks.
        int k_VKQ_0 = GID_1 * NBATCH_FA;
        while (k_VKQ_0 < k_VKQ_max - NBATCH_FA) {
            const int oob_check = 0;
            fattn_tile_iter(
                oob_check,
                Q_tmp, 
                K_h2, 
                V_h2, 
                maskh,
                Q_D2,
                logit_softcap, 
                slope, 
                KQ, 
                KV_tmp,
                stride_K2, 
                stride_V2, 
                stride_mask, 
                KQ_max, 
                KQ_sum, 
                VKQ, 
                k_VKQ_0, 
                k_VKQ_max, 
                col_Q_0,
                KQ_max_new_shared);
            k_VKQ_0 += GDIM_1 * NBATCH_FA;
        }
        if (k_VKQ_0 < k_VKQ_max) {
            const int oob_check = 1;
            fattn_tile_iter(
                oob_check,
                Q_tmp, 
                K_h2, 
                V_h2, 
                maskh,
                Q_D2,
                logit_softcap, 
                slope, 
                KQ, 
                KV_tmp,
                stride_K2, 
                stride_V2, 
                stride_mask, 
                KQ_max, 
                KQ_sum, 
                VKQ, 
                k_VKQ_0, 
                k_VKQ_max, 
                col_Q_0,
                KQ_max_new_shared);
        }
    } else {
        // Branch without out-of-bounds checks.
        for (int k_VKQ_0 = GID_1 * NBATCH_FA; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += GDIM_1 * NBATCH_FA) {
            const bool oob_check = 0;
            fattn_tile_iter(
                oob_check,
                Q_tmp, 
                K_h2, 
                V_h2, 
                maskh, 
                Q_D2,
                logit_softcap, 
                slope, 
                KQ, 
                KV_tmp,
                stride_K2, 
                stride_V2, 
                stride_mask, 
                KQ_max, 
                KQ_sum, 
                VKQ, 
                k_VKQ_0, 
                k_VKQ_max, 
                col_Q_0,
                KQ_max_new_shared);
        }
    }

    unroll_for (int jc0 = 0; jc0 < CPSG; jc0++) {
        KQ_sum[jc0] = sub_group_reduce_add(KQ_sum[jc0]);
    }
 
    if (NP > 1) {
        // reuse local memory
        local half2 *VKQ_combine = (local half2 *)KV_tmp;
        local float *KQ_sum_combine = (local float *)Q_tmp;

        if (LID_1 % NP != 0) {
            unroll_for (int i0 = 0; i0 < DV_P / 2; i0 += SG_SIZE * CPY_NE_DV) {
                COPY_LR_H2(
                    CPY_NE_DV,
                    &VKQ_combine[
                        LID_1 * (DV_P / 2) + 
                        i0 + 
                        LID_0 * CPY_NE_DV], 
                    &VKQ[i0 / SG_SIZE]);
            }
            if (LID_0 == 0) {
                KQ_sum_combine[LID_1] = KQ_sum[0];
            }
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);

        if (LID_1 % NP != 0) {
            return;
        }

        unroll_for (int ip = 1; ip < NP; ip++) {
            unroll_for (int i0 = 0; i0 < DV_P / 2; i0 += SG_SIZE * CPY_NE_DV) {
                half2 tmp[CPY_NE_DV] ALIGNED;
                COPY_RL_H2(
                    CPY_NE_DV,
                    tmp, 
                    &VKQ_combine[
                        (LID_1 + ip) * (DV_P / 2) + 
                        i0 + 
                        LID_0 * CPY_NE_DV]);
                unroll_for (int i1 = 0; i1 < CPY_NE_DV; i1++) {
                    VKQ[i0 / SG_SIZE + i1] += tmp[i1];
                }
            }

            KQ_sum[0] += KQ_sum_combine[LID_1 + ip];
        }
    }

    // Here LID_1 % NP == 0 (see return above)
 
    // Attention sink: adjust KQ max and sum only for the first of all parallel work-groups
    if (sinks && GID_1 == 0) {
        unroll_for (int jc0 = 0; jc0 < CPSG; jc0++) {
            const int jc = jc0 + (LID_1 / NP) * CPSG;
            const float sink = sinks[head0 + jc % NCOLS2];

            float KQ_max_new_j = fmax(KQ_max[jc0], sink);
            const float KQ_max_scale = exp(KQ_max[jc0] - KQ_max_new_j);
            KQ_max[jc0] = KQ_max_new_j;

            const float val = exp(sink - KQ_max[jc0]);
            KQ_sum[jc0] = KQ_sum[jc0] * KQ_max_scale + val;

            const half2 KQ_max_scale_h2 = (half2)(KQ_max_scale, KQ_max_scale);
            unroll_for (int i0 = 0; i0 < DV_P / 2; i0 += SG_SIZE) {
                VKQ[jc0 * ((DV_P / 2) / SG_SIZE) + i0 / SG_SIZE] *= KQ_max_scale_h2;
            }
        }
    }
 
    // Write back results
    unroll_for (int jc0 = 0; jc0 < CPSG; jc0++) {
        const int jc = jc0 + (LID_1 / NP) * CPSG;

        const int j = jc / NCOLS2;
        const int c = jc % NCOLS2;

        if (NCOLS1 > 1 && col_Q_0 + j >= Q_D2) {
            return;
        }

        const float scale = (GDIM_1 == 1) ? 1.0f / KQ_sum[jc0] : 1.0f;

        const int j_dst_unrolled = 
            ((sequence * Q_D2 + col_Q_0 + j) * Q_D1 + head0 + c) * GDIM_1 + GID_1;

        unroll_for (int i0 = 0; i0 < DV_P / 2; i0 += SG_SIZE * CPY_NE_DV2) {
            float2 tmp[CPY_NE_DV2] ALIGNED;
            unroll_for (int i1 = 0; i1 < CPY_NE_DV2; i1++) {
                tmp[i1] = convert_float2(VKQ[jc0 * ((DV_P / 2) / SG_SIZE) + i0 / SG_SIZE + i1]);
                tmp[i1].x *= scale;
                tmp[i1].y *= scale;
            }
            if (i0 + SG_SIZE * CPY_NE_DV2 <= DV / 2 || i0 + LID_0 * CPY_NE_DV2 < DV / 2) {
                COPY_GR_F2(
                    CPY_NE_DV2,
                    &dst[
                        j_dst_unrolled * DV + 
                        2 * i0 + 
                        LID_0 * (2 * CPY_NE_DV2)],
                    tmp);
            }
        }

        if (GDIM_1 != 1 && LID_0 == 0) {
            dst_meta[j_dst_unrolled] = (float2)(KQ_max[jc0], KQ_sum[jc0]);
        }
    } 
}

)";

} // namespace

const char *FattnTileKernelCode() {
    return g_kernelCodeFattn;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

