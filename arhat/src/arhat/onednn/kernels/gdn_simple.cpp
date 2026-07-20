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

const char g_kernelCodeGdn[] = R"(
__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void gdn_simple(
        const global float *q,
        const global float *k,
        const global float *v,
        const global float *g,
        const global float *beta,
        const global float *curr_state,
        global float *dst,
        float scale
        SHAPE_INFO_ARGS) {

    q += Q_BASE;
    k += K_BASE;
    v += V_BASE;
    g += G_BASE;
    beta += BETA_BASE;
    curr_state += STATE_BASE;
    dst += DST_BASE;

    const int h_idx = GID_0;
    const int sequence = GID_1;
    // each subgroup owns one column, using subgroup primitives to reduce across rows 
    const int lane = LID_0;
    const int col = GID_2 * LDIM_1 + LID_1;

    const int RQ0 = V_D0 / Q_D0;
    const int iq2 = h_idx % Q_D2;
    const int iq0 = sequence / RQ0;

    const long attn_score_elems = S_V * H * N_TOKENS * N_SEQS;
    global float *attn_data = dst;
    global float *state = dst + attn_score_elems;

    const long state_offset = (sequence * H + h_idx) * S_V * S_V;
    state += state_offset;
    curr_state += state_offset + col * S_V;
    attn_data += (sequence * N_TOKENS * H + h_idx) * S_V;

    float s_shard[ROWS_PER_LANE];

    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous 
    unroll_for (int r = 0; r < ROWS_PER_LANE; r++) {
        const int i = r * SG_SIZE + lane;
        s_shard[r] = curr_state[i];
    }

    for (int t = 0; t < N_TOKENS; t++) {
        const global float *q_t = q + iq0 * Q_S0 + t * Q_S1 + iq2 * Q_S2;
        const global float *k_t = k + iq0 * Q_S0 + t * Q_S1 + iq2 * Q_S2;
        const global float *v_t = v + sequence * V_S0 + t * V_S1 + h_idx * V_S2;

        const long gb_offset = sequence * BETA_S0 + t * BETA_S1 + h_idx * BETA_S2;
        const global float *beta_t = beta + gb_offset;
        const global float *g_t = g + gb_offset * (KDA ? S_V : 1);

        const float beta_val = *beta_t;
 
        // Cache k and q in registers
        float k_reg[ROWS_PER_LANE];
        float q_reg[ROWS_PER_LANE];
        unroll_for (int r = 0; r < ROWS_PER_LANE; r++) {
            const int i = r * SG_SIZE + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }
 
        if (!KDA) {
            const float g_val = exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
            unroll_for (int r = 0; r < ROWS_PER_LANE; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = sub_group_reduce_add(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
            unroll_for (int r = 0; r < ROWS_PER_LANE; r++) {
                s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = sub_group_reduce_add(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else { 
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
            unroll_for (int r = 0; r < ROWS_PER_LANE; r++) {
                const int i = r * SG_SIZE + lane;
                kv_shard += exp(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = sub_group_reduce_add(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
            unroll_for (int r = 0; r < ROWS_PER_LANE; r++) {
                const int i = r * SG_SIZE + lane;
                s_shard[r] = exp(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = sub_group_reduce_add(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_V * H;
    } 

    // Write state back to global memory (transposed layout)
    unroll_for (int r = 0; r < ROWS_PER_LANE; r++) {
        const int i = r * SG_SIZE + lane;
        state[col * S_V + i] = s_shard[r];
    }
}

)";

} // namespace

const char *GdnSimpleKernelCode() {
    return g_kernelCodeGdn;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

