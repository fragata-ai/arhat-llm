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
inline float get_alibi_slope(
        const float max_bias, 
        const uint h, 
        const float m0, 
        const float m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = (h < N_HEAD_LOG2) ? m0 : m1;
    const int exph = (h < N_HEAD_LOG2) ? h + 1 : 2 * (h - N_HEAD_LOG2) + 1;
    return pow(base, exph);
}

__kernel void fattn_simple(
        const global Q_TYPE *q_data,
        const global K_TYPE *k_data,
        const global V_TYPE *v_data,
        const global MASK_TYPE *mask_data,
        const global SINKS_TYPE *sinks_data,
        global O_TYPE *o_data,
        const float scale,
        const float max_bias,
        const float logit_softcap,
        const float m0,
        const float m1) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / N_HEAD;
    const int head_idx = head_batch_idx % N_HEAD;

    const int gqa_ratio = N_HEAD / N_HEAD_KV;
    const int head_kv_idx = head_idx / gqa_ratio;

    q_data += Q_BASE;
    k_data += K_BASE;
    v_data += V_BASE;
    o_data += O_BASE;

    const global MASK_TYPE *mask_base = NULL;
    if (mask_data != NULL) {
        const int mask_batch_idx = batch_idx % MASK_D0;
        const int mask_head_idx = head_idx % MASK_D1;
        mask_base = mask_data + MASK_BASE + mask_batch_idx * MASK_S0 + mask_head_idx * MASK_S1;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < N_Q) {
        const ulong q_row_offset = batch_idx * Q_S0 + head_idx * Q_S1 + my_query_row * Q_S2;
        const global Q_TYPE4 *q_ptr = (const global Q_TYPE4 *)(q_data + q_row_offset);
        unroll_for (int i = 0; i < DK_VEC; i++) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
        }
    }

    ACC_TYPE4 o_acc[DV_VEC];
    unroll_for (int i = 0; i < DV_VEC; i++) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, m0, m1);

    __local K_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local V_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < N_KV; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < N_KV) {
                const ulong k_row_offset = batch_idx * K_S0 + head_kv_idx * K_S1 + k_row_idx * K_S2;
                l_k[row][col] = ((global K_TYPE4 *)(k_data + k_row_offset))[col];
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < N_KV) {
                const ulong v_row_offset = batch_idx * V_S0 + head_kv_idx * V_S1 + v_row_idx * V_S2;
                l_v[row][col] = ((global V_TYPE4 *)(v_data + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row < N_Q) {
            for (int j = 0; j < BLOCK_N; j += 2) {
                const int k_row0 = k_start + j;
                const int k_row1 = k_start + j + 1;

                ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
                ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
                unroll_for (int k = 0; k < DK_VEC; k++) {
                    dot_acc0 = mad(q_priv[k], CONVERT_K_ACC4(l_k[j][k]), dot_acc0);
                    dot_acc1 = mad(q_priv[k], CONVERT_K_ACC4(l_k[j + 1][k]), dot_acc1);
                }
                ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
                ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

                if (IS_CAUSAL) {
                    if (k_row0 > N_KV - N_Q + my_query_row) {
                        score0 = -INFINITY;
                    }
                    if (k_row1 > N_KV - N_Q + my_query_row) {
                        score1 = -INFINITY;
                    }
                }

                if (k_row0 >= N_KV) {
                    score0 = -INFINITY;
                }
                if (k_row1 >= N_KV) {
                    score1 = -INFINITY;
                }

                if (mask_base != NULL) {
                    const global MASK_TYPE *mask_ptr = mask_base + my_query_row * MASK_S2;
                    if (k_row0 < N_KV) {
                        score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                    }
                    if (k_row1 < N_KV) {
                        score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
                    }
                }

                if (logit_softcap > 0.0f) {
                    score0 = logit_softcap * tanh(score0 / logit_softcap);
                    score1 = logit_softcap * tanh(score1 / logit_softcap);
                }

                const ACC_TYPE m_new = max(m_i, max(score0, score1));
                if (!isinf(m_new)) {
                    const ACC_TYPE p0 = exp(score0 - m_new);
                    const ACC_TYPE p1 = exp(score1 - m_new);
                    const ACC_TYPE scale_prev = exp(m_i - m_new);

                    unroll_for (int i = 0; i < DV_VEC; i++) {
                        o_acc[i] = o_acc[i] * scale_prev + 
                            p0 * CONVERT_V_ACC4(l_v[j][i]) + p1 * CONVERT_V_ACC4(l_v[j + 1][i]);
                    }
                    l_i = l_i * scale_prev + p0 + p1;
                    m_i = m_new;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (my_query_row < N_Q) {
        if (sinks_data != NULL) {
            const ACC_TYPE m_sink = (ACC_TYPE)sinks_data[SINKS_BASE + head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);
            const ACC_TYPE scale_o = exp(m_i - m_final);
            unroll_for (int i = 0; i < DV_VEC; i++) {
                o_acc[i] *= scale_o;
            }
            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * O_S0 + my_query_row * O_S1 + head_idx * O_S2;
        global O_TYPE4 *o_row = (global O_TYPE4 *)(o_data + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            unroll_for (int i = 0; i < DV_VEC; i++) {
                o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            unroll_for (int i = 0; i < DV_VEC; i++) {
                o_row[i] = (O_TYPE4)(0.0f);
            }
        }
    }
}

)";

const char g_kernelCodeFattnQ1[] = R"(
inline float get_alibi_slope(
        const float max_bias, 
        const uint h, 
        const float m0, 
        const float m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = (h < N_HEAD_LOG2) ? m0 : m1;
    const int exph = (h < N_HEAD_LOG2) ? h + 1 : 2 * (h - N_HEAD_LOG2) + 1;
    return pow(base, exph);
}

__kernel void fattn_simple(
        const global Q_TYPE *q_data,
        const global K_TYPE *k_data,
        const global V_TYPE *v_data,
        const global MASK_TYPE *mask_data,
        const global SINKS_TYPE *sinks_data,
        global O_TYPE *o_data,
        const float scale,
        const float max_bias,
        const float logit_softcap,
        const float m0,
        const float m1) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / N_HEAD;
    const int head_idx = head_batch_idx % N_HEAD;

    const int gqa_ratio = N_HEAD / N_HEAD_KV;
    const int head_kv_idx = head_idx / gqa_ratio;

    q_data += Q_BASE;
    k_data += K_BASE;
    v_data += V_BASE;
    o_data += O_BASE;

    const global MASK_TYPE *mask_base = NULL;
    if (mask_data != NULL) {
        const int mask_batch_idx = batch_idx % MASK_D0;
        const int mask_head_idx = head_idx % MASK_D1;
        mask_base = mask_data + MASK_BASE + mask_batch_idx * MASK_S0 + mask_head_idx * MASK_S1;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * Q_S0 + head_idx * Q_S1;
    const global Q_TYPE4 *q_ptr = (const global Q_TYPE4 *)(q_data + q_row_offset);
    unroll_for (int i = 0; i < DK_VEC; i++) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, m0, m1);

    ACC_TYPE m_i = (sinks_data != NULL) ? sinks_data[SINKS_BASE + head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < N_KV; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * K_S0 + head_kv_idx * K_S1 + k_idx * K_S2;
        const global K_TYPE4 *k_ptr = (const global K_TYPE4 *)(k_data + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        unroll_for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_K_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            score += slope * (ACC_TYPE)mask_base[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    __local ACC_TYPE local_m[Q1_WG_SIZE];

    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);

    unroll_for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_m[tid] = max(local_m[tid], local_m[tid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    ACC_TYPE4 o_acc[DV_VEC];
    unroll_for (int i = 0; i < DV_VEC; i++) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < N_KV; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * K_S0 + head_kv_idx * K_S1 + k_idx * K_S2;
        const ulong v_row_offset = batch_idx * V_S0 + head_kv_idx * V_S1 + k_idx * V_S2;
        const global K_TYPE4 *k_ptr = (const global K_TYPE4 *)(k_data + k_row_offset);
        const global V_TYPE4 *v_ptr = (const global V_TYPE4 *)(v_data + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        unroll_for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_K_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            score += slope * (ACC_TYPE)mask_base[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_final);
        l_i += p;
        unroll_for (int i = 0; i < DV_VEC; i++) {
            o_acc[i] = mad(p, CONVERT_V_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];

    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);

    unroll_for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_l[tid] += local_l[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong o_row_offset = batch_idx * O_S0 + head_idx * O_S2;
    global O_TYPE4 *o_row = (global O_TYPE4 *)(o_data + o_row_offset);
    ACC_TYPE l_final = local_l[0];

    if (sinks_data != NULL) {
        l_final += exp(sinks_data[SINKS_BASE + head_idx] - m_final);
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            unroll_for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    local_o_comp[tid] += local_o_comp[tid + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        unroll_for (int i = 0; i < DV_VEC; i++) {
            o_row[i] = (O_TYPE4)(0.0f);
        }
    }
} 

)";

} // namespace

const char *FattnSimpleKernelCode() {
    return g_kernelCodeFattn;
}

const char *FattnSimpleQ1KernelCode() {
    return g_kernelCodeFattnQ1;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

