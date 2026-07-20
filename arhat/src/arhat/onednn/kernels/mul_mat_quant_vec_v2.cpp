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

//
//    MulMatVecV2Base
//

const char g_codeMmvqBase[] = R"(
#ifndef MUL_MAT_ID
inline uint3 get_offsets(
        const uint batch_stride_a,
        const uint batch_stride_b,
        const uint batch_stride_d, 
        const uint base_work_group_y,
        const uint a_dim1,
        const uint b_dim1,
        const uint a_bcast0,
        const uint a_bcast1) {

    const uint batch_idx = GID_1 + base_work_group_y;

    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint b_idx0 = batch_idx / b_dim1;
        const uint b_idx1 = batch_idx % b_dim1;

        const uint a_idx0 = b_idx0 / a_bcast0;
        const uint a_idx1 = b_idx1 / a_bcast1;

        batch_idx_a = a_idx0 * a_dim1 + a_idx1;
    }

    const uint a_offset = batch_idx_a * (batch_stride_a / QUANT_K);
    const uint b_offset = batch_idx * batch_stride_b;
    const uint d_offset = batch_idx * batch_stride_d;

    return (uint3)(a_offset, b_offset, d_offset);
} 

#else
inline uint3 get_offsets(
        const global int *data_ids,
        const uint stride_b,
        const uint stride_d,
        const uint batch_stride_a,
        const uint batch_stride_b,
        const uint batch_stride_d,
        const uint b_dim2,
        const uint ids_i2,
        const uint ids_stride2) {

    const uint ids_i3 = GID_1;

    const uint expert_id = data_ids[ids_i2 * ids_stride2 + ids_i3];

    const uint a_offset = expert_id * (batch_stride_a / QUANT_K);
    const uint b_offset = ids_i2 * batch_stride_b + (ids_i3 % b_dim2) * stride_b;
    const uint d_offset = ids_i2 * batch_stride_d + ids_i3 * stride_d;

    return (uint3)(a_offset, b_offset, d_offset);
}

#endif

#ifdef USE_SUBGROUP_ADD_NO_SHMEM
void reduce_result(
        global D_TYPE *data_d,
        const uint batch_stride_d,
        FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], 
        const uint d_offset, 
        const uint first_row, 
        const uint num_rows, 
        const uint tid) {

    unroll_for (uint j = 0; j < NUM_COLS; j++) {
        unroll_for (uint n = 0; n < num_rows; n++) {
            temp[j][n] = sub_group_reduce_add(temp[j][n]);
        }
    }

    if (tid == 0) {
        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            unroll_for (uint n = 0; n < num_rows; n++) {
                data_d[j * batch_stride_d + d_offset + first_row + n] = (D_TYPE)temp[j][n];
            }
        }
    }
}

#else

void reduce_result(
        global D_TYPE *data_d,
        const uint batch_stride_d,
        FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], 
        local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
        const uint d_offset, 
        const uint first_row, 
        const uint num_rows, 
        const uint tid) {

    // sub_group_reduce_add is probably faster on devices that support it,
    // particularly when the workgroup has more than one subgroup

#if USE_SUBGROUP_ADD
    // sum up partial sums within a subgroup
    unroll_for (uint j = 0; j < NUM_COLS; j++) {
        unroll_for (uint n = 0; n < num_rows; n++) {
            temp[j][n] = sub_group_reduce_add(temp[j][n]);
        }
    }

    // Go through shared memory to sum partials across subgroups
    if (get_sub_group_local_id() == 0) {
        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            unroll_for (uint n = 0; n < num_rows; n++) {
                tmpsh[j][n][get_sub_group_id()] = temp[j][n];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            unroll_for (uint n = 0; n < num_rows; n++) {
                temp[j][n] = (FLOAT_TYPE)0.0f;
                unroll_for (uint s = 0; s < get_num_sub_groups(); s++) {
                    temp[j][n] += tmpsh[j][n][s];
                }
                data_d[j * batch_stride_d + d_offset + first_row + n] = (D_TYPE)temp[j][n];
            }
        }
    }

#else
    // sum up partial sums and write back result
    unroll_for (uint j = 0; j < NUM_COLS; j++) {
        unroll_for (uint n = 0; n < num_rows; n++) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unroll_for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            unroll_for (uint j = 0; j < NUM_COLS; j++) {
                unroll_for (uint n = 0; n < num_rows; n++) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            unroll_for (uint n = 0; n < num_rows; n++) {
                data_d[j * batch_stride_d + d_offset + first_row + n] = (D_TYPE)tmpsh[j][n][0];
            }
        }
    }
#endif
}

#endif 

)";

//
//    MulMatQuantVecV2Defs
//

const char g_codeMmvqDefs_Q4_0[] = R"(
#define QUANT_K_Q4_0 32
#define QUANT_R_Q4_0 2

typedef struct {
    half d;
    uchar qs[16];
} block_q4_0;

typedef struct {
    half d;
    ushort qs[16 / 2];
} block_q4_0_packed16;

#define QUANT_K QUANT_K_Q4_0
#define QUANT_R QUANT_R_Q4_0
#define QUANT_AUXF 1
#define A_TYPE block_q4_0
#define A_TYPE_PACKED16 block_q4_0_packed16
#define DATA_A_QUANT_LEGACY

)";

const char g_codeMmvqDefs_Q4_1[] = R"(
#define QUANT_K_Q4_1 32
#define QUANT_R_Q4_1 2

typedef struct {
    half d;
    half m;
    uchar qs[16];
} block_q4_1;

typedef struct {
    half d;
    half m;
    ushort qs[16 / 2];
} block_q4_1_packed16;

typedef struct {
    half2 dm;
    uint qs[16 / 4];
} block_q4_1_packed32;

#define QUANT_K QUANT_K_Q4_1
#define QUANT_R QUANT_R_Q4_1
#define QUANT_AUXF 2
#define A_TYPE block_q4_1
#define A_TYPE_PACKED16 block_q4_1_packed16
#define A_TYPE_PACKED32 block_q4_1_packed32
#define DATA_A_QUANT_LEGACY

)";

const char g_codeMmvqDefs_Q5_0[] = R"(
#define QUANT_K_Q5_0 32
#define QUANT_R_Q5_0 2

typedef struct {
    half d;
    ushort qh[2];
    uchar qs[16];
} block_q5_0;

typedef struct {
    half d;
    ushort qh[2];
    ushort qs[16 / 2];
} block_q5_0_packed16;

#define QUANT_K QUANT_K_Q5_0
#define QUANT_R QUANT_R_Q5_0
#define QUANT_AUXF 1
#define A_TYPE block_q5_0
#define A_TYPE_PACKED16 block_q5_0_packed16
#define DATA_A_QUANT_LEGACY

)";

const char g_codeMmvqDefs_Q5_1[] = R"(
#define QUANT_K_Q5_1 32
#define QUANT_R_Q5_1 2

typedef struct {
    half d;
    half m;
    uint qh;
    uchar qs[16];
} block_q5_1;

typedef struct {
    half d;
    half m;
    uint qh;
    ushort qs[16 / 2];
} block_q5_1_packed16;

typedef struct {
    half2 dm;
    uint qh;
    uint qs[16 / 4];
} block_q5_1_packed32;

#define QUANT_K QUANT_K_Q5_1
#define QUANT_R QUANT_R_Q5_1
#define QUANT_AUXF 2
#define A_TYPE block_q5_1
#define A_TYPE_PACKED16 block_q5_1_packed16
#define A_TYPE_PACKED32 block_q5_1_packed32
#define DATA_A_QUANT_LEGACY

)";

const char g_codeMmvqDefs_Q8_0[] = R"(
#define QUANT_K_Q8_0 32
#define QUANT_R_Q8_0 1

typedef struct {
    half d;
    char qs[32];
} block_q8_0;

typedef struct {
    half d;
    short qs[32 / 2];
} block_q8_0_packed16;

#define QUANT_K QUANT_K_Q8_0
#define QUANT_R QUANT_R_Q8_0
#define QUANT_AUXF 1
#define A_TYPE block_q8_0
#define A_TYPE_PACKED16 block_q8_0_packed16
#define DATA_A_QUANT_LEGACY

)";

const char g_codeMmvqDefs_Q8_1[] = R"(
#define QUANT_K_Q8_1 32
#define QUANT_R_Q8_1 1

typedef struct {
    half2 ds;
    char qs[32];
} block_q8_1;

typedef struct {
    half2 ds;
    short qs[16];
} block_q8_1_packed16;

typedef struct {
    half2 ds;
    int qs[8];
} block_q8_1_packed32;

// 4 blocks in one to allow 16-byte/128-bit alignment and loads
typedef struct {
    half2 ds[4];
    int qs[32];
} block_q8_1_x4;

typedef struct {
    half2 ds[4];
    int4 qs[8];
} block_q8_1_x4_packed128;

)";

const char g_codeMmvqDefs_Q2_K[] = R"(
// K-quants
#define QUANT_K_Q2_K 256

typedef struct {
    uchar scales[QUANT_K_Q2_K / 16];
    uchar qs[QUANT_K_Q2_K / 4];
    half2 dm;
} block_q2_K;

typedef struct {
    ushort scales[QUANT_K_Q2_K / 16 / 2];
    ushort qs[QUANT_K_Q2_K / 4 / 2];
    half2 dm;
} block_q2_K_packed16;

typedef struct {
    uint scales[QUANT_K_Q2_K / 16 / 4];
    uint qs[QUANT_K_Q2_K / 4 / 4];
    half2 dm;
} block_q2_K_packed32;

#define QUANT_K QUANT_K_Q2_K
#define QUANT_R 1
#define A_TYPE block_q2_K
#define A_TYPE_PACKED16 block_q2_K_packed16
#define A_TYPE_PACKED32 block_q2_K_packed32
#define SCALES_PER_32 2
#define DATA_A_QUANT_K

)";

const char g_codeMmvqDefs_Q3_K[] = R"(
#define QUANT_K_Q3_K 256

typedef struct {
    uchar hmask[QUANT_K_Q3_K/8];
    uchar qs[QUANT_K_Q3_K/4];
    uchar scales[12];
    half d;
} block_q3_K;

typedef struct {
    ushort hmask[QUANT_K_Q3_K / 8 / 2];
    ushort qs[QUANT_K_Q3_K / 4 / 2];
    ushort scales[12 / 2];
    half d;
} block_q3_K_packed16;

#define QUANT_K QUANT_K_Q3_K
#define QUANT_R 1
#define A_TYPE block_q3_K
#define A_TYPE_PACKED16 block_q3_K_packed16
#define DATA_A_QUANT_K

)";

const char g_codeMmvqDefs_Q4_K[] = R"(
#define QUANT_K_Q4_K 256

typedef struct {
    half2 dm;
    uchar scales[3 * QUANT_K_Q4_K / 64];
    uchar qs[QUANT_K_Q4_K / 2];
} block_q4_K;

typedef struct {
    half2 dm;
    ushort scales[3 * QUANT_K_Q4_K / 64 / 2];
    ushort qs[QUANT_K_Q4_K / 2 / 2];
} block_q4_K_packed16;

typedef struct {
    half2 dm;
    uint scales[3 * QUANT_K_Q4_K / 64 / 4];
    uint qs[QUANT_K_Q4_K / 2 / 4];
} block_q4_K_packed32;

typedef struct {
    uint4 q4k[9];
} block_q4_K_packed128;

#define QUANT_K QUANT_K_Q4_K
#define QUANT_R 1
#define A_TYPE block_q4_K
#define A_TYPE_PACKED16 block_q4_K_packed16
#define A_TYPE_PACKED32 block_q4_K_packed32
#define DATA_A_QUANT_K

)";

const char g_codeMmvqDefs_Q5_K[] = R"(
#define QUANT_K_Q5_K 256

typedef struct {
    half2 dm;
    uchar scales[12];
    uchar qh[QUANT_K_Q5_K / 8];
    uchar qs[QUANT_K_Q5_K / 2];
} block_q5_K;

typedef struct {
    half2 dm;
    ushort scales[12 / 2];
    ushort qh[QUANT_K_Q5_K / 8 / 2];
    ushort qs[QUANT_K_Q5_K / 2 / 2];
} block_q5_K_packed16;

typedef struct {
    half2 dm;
    uint scales[12 / 4];
    uint qh[QUANT_K_Q5_K / 8 / 4];
    uint qs[QUANT_K_Q5_K / 2 / 4];
} block_q5_K_packed32;

typedef struct {
    uint4 q5k[11];
} block_q5_K_packed128;

#define QUANT_K QUANT_K_Q5_K
#define QUANT_R 1
#define A_TYPE block_q5_K
#define A_TYPE_PACKED16 block_q5_K_packed16
#define A_TYPE_PACKED32 block_q5_K_packed32
#define DATA_A_QUANT_K

)";

const char g_codeMmvqDefs_Q6_K[] = R"(
#define QUANT_K_Q6_K 256

typedef struct {
    uchar ql[QUANT_K_Q6_K / 2];
    uchar qh[QUANT_K_Q6_K / 4];
    char scales[QUANT_K_Q6_K / 16];
    half d;
} block_q6_K;

typedef struct {
    ushort ql[QUANT_K_Q6_K / 2 / 2];
    ushort qh[QUANT_K_Q6_K / 4 / 2];
    short scales[QUANT_K_Q6_K / 16 / 2];
    half d;
} block_q6_K_packed16;

#define QUANT_K QUANT_K_Q6_K
#define QUANT_R 1
#define A_TYPE block_q6_K
#define A_TYPE_PACKED16 block_q6_K_packed16
#define DATA_A_QUANT_K

)";

const char g_codeMmvqDefs_Mxfp4[] = R"(
#define QUANT_K_MXFP4 32
#define QUANT_R_MXFP4 2

typedef struct {
    uchar e;
    uchar qs[QUANT_K_MXFP4 / 2];
} block_mxfp4;

#define QUANT_K QUANT_K_MXFP4
#define QUANT_R QUANT_R_MXFP4
#define QUANT_AUXF 1
#define A_TYPE block_mxfp4

)";

//
//    MulMatQuantVecV2Impl
//

const char g_codeMmvqImpl_Q4_0[] = R"(
inline float get_dm(const global block_q4_0 *data_a, const uint ib) {
    return (float)data_a[ib].d;
} 

// Each iqs value maps to a 32-bit integer
// 2-byte loads for Q4_0 blocks (18 bytes)
inline int2 repack(
        const global block_q4_0 *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q4_0_packed16 *data_a_packed16 = (const global block_q4_0_packed16 *)data_a;
    const ushort2 quants = 
        (ushort2)(data_a_packed16[ib].qs[iqs * 2], data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint vui = as_uint(quants);
    return (int2)(vui & 0x0F0F0F0F, (vui >> 4) & 0x0F0F0F0F);
}

inline float mul_q8_1(
        const int q_sum, 
        const float da, 
        const float2 dsb, 
        const int sum_divisor) {
    return da * ((float)q_sum * dsb.x - (8 / sum_divisor) * dsb.y);
} 

)";

const char g_codeMmvqImpl_Q4_1[] = R"(
inline float2 get_dm(const global block_q4_1 *data_a, const uint ib) {
    const global block_q4_1_packed32 *data_a_packed32 = (const global block_q4_1_packed32 *)data_a;
    return convert_float2(data_a_packed32[ib].dm);
} 

// 4-byte loads for Q4_1 blocks (20 bytes)
inline int2 repack(
        const global block_q4_1 *data_a,
        const uint ib, 
        const uint iqs) {
    const global block_q4_1_packed32 *data_a_packed32 = (const global block_q4_1_packed32 *)data_a;
    const uint vui = data_a_packed32[ib].qs[iqs];
    return (int2)(vui & 0x0F0F0F0F, (vui >> 4) & 0x0F0F0F0F);
}

inline float mul_q8_1(
        const int q_sum, 
        const float2 dma, 
        const float2 dsb, 
        const int sum_divisor) {
    return (float)q_sum * dma.x * dsb.x + dma.y * dsb.y / sum_divisor;
} 

)";

const char g_codeMmvqImpl_Q5_0[] = R"(
inline float get_dm(const global block_q5_0 *data_a, const uint ib) {
    return (float)data_a[ib].d;
} 

// 2-byte loads for Q5_0 blocks (22 bytes)
inline int2 repack(
        const global block_q5_0 *data_a,
        const uint ib, 
        const uint iqs) {
    const global block_q5_0_packed16 *data_a_packed16 = (const global block_q5_0_packed16 *)data_a;
    const ushort2 quants = 
        (ushort2)(data_a_packed16[ib].qs[iqs * 2], data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint vui = as_uint(quants);
    const int qh = 
        (int)(((uint)data_a_packed16[ib].qh[1] << 16 | data_a_packed16[ib].qh[0]) >> (4 * iqs));
    const int v0 = 
        (int)(vui & 0x0F0F0F0F) | 
            ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)
    const int v1 = 
        (int)((vui >> 4) & 0x0F0F0F0F) | 
            (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)
    return (int2)(v0, v1);
}

inline float mul_q8_1(
        const int q_sum, 
        const float da, 
        const float2 dsb, 
        const int sum_divisor) {
    return da * ((float)q_sum * dsb.x - (16 / sum_divisor) * dsb.y);
} 

)";

const char g_codeMmvqImpl_Q5_1[] = R"(
inline float2 get_dm(const global block_q5_1 *data_a, const uint ib) {
    const global block_q5_1_packed32 *data_a_packed32 = (const global block_q5_1_packed32 *)data_a;
    return convert_float2(data_a_packed32[ib].dm);
} 

// 4-byte loads for Q5_1 blocks (24 bytes)
inline int2 repack(
        const global block_q5_1 *data_a,
        const uint ib, 
        const uint iqs) {
    const global block_q5_1_packed16 *data_a_packed16 = (const global block_q5_1_packed16 *)data_a;
    const global block_q5_1_packed32 *data_a_packed32 = (const global block_q5_1_packed32 *)data_a;
    const ushort2 quants = 
        (ushort2)(data_a_packed16[ib].qs[iqs * 2], data_a_packed16[ib].qs[iqs * 2 + 1]);
    const uint vui = as_uint(quants);
    const int qh = (int)(data_a_packed32[ib].qh >> (4 * iqs));
    const int v0 = 
        (int)(vui & 0x0F0F0F0F) | 
            ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)
    const int v1 = 
        (int)((vui >> 4) & 0x0F0F0F0F) | 
            (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)
    return (int2)(v0, v1);
}

inline float mul_q8_1(
        const int q_sum, 
        const float2 dma, 
        const float2 dsb, 
        const int sum_divisor) {
    return (float)q_sum * dma.x * dsb.x + dma.y * dsb.y / sum_divisor;
} 

)";

const char g_codeMmvqImpl_Q8_0[] = R"(
inline float get_dm(const global block_q8_0 *data_a, const uint ib) {
    return (float)data_a[ib].d;
} 

// 2-byte loads for Q8_0 blocks (34 bytes)
inline int repack(
        const global block_q8_0 *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q8_0_packed16 *data_a_packed16 = (const global block_q8_0_packed16 *)data_a;
    return as_int((short2)(
        data_a_packed16[ib].qs[iqs * 2],
        data_a_packed16[ib].qs[iqs * 2 + 1]));
}

inline float mul_q8_1(
        const int q_sum, 
        const float da, 
        const float2 dsb, 
        const int sum_divisor) {
    return (float)q_sum * da * dsb.x;
} 

)";

const char g_codeMmvqImplLegacy[] = R"(
// Common code for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and MXFP4

inline float mmvq_dot_product(
        const global A_TYPE *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    int q_sum = 0;
#if QUANT_R == 2
    const int2 data_a_qs = repack(data_a, ib_a, iqs);
    q_sum = IMAD(data_a_qs.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(data_a_qs.y, cache_b_qs[1], q_sum);
#else
    int data_a_qs = repack(data_a, ib_a, iqs * 2);
    q_sum = IMAD(data_a_qs, cache_b_qs[0], q_sum);
    data_a_qs = repack(data_a, ib_a, iqs * 2 + 1);
    q_sum = IMAD(data_a_qs, cache_b_qs[1], q_sum);
#endif

    // 2 quants per call => divide sums by 8/2 = 4
    return mul_q8_1(q_sum, get_dm(data_a, ib_a), cache_b_ds, 4);
}

)";

const char g_codeMmvqImpl_Q2_K[] = R"(
inline float2 get_dm(const global block_q2_K *data_a, const uint ib) {
    const global block_q2_K_packed32 *data_a_packed32 = (const global block_q2_K_packed32 *)data_a;
    const uint ib_k = ib / 8;
    return convert_float2(data_a_packed32[ib_k].dm);
} 

// 4-byte loads for Q2_K blocks (84 bytes)
inline int4 repack4(
        const global block_q2_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q2_K_packed32 *data_a_packed32 = (const global block_q2_K_packed32 *)data_a;

    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    return (int4)(
        (data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x03030303,
        (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303,
        (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303,
        (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303);
}

inline uchar get_scale(
        const global block_q2_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    return data_a[ib_k].scales[iqs_k / 4];
}

float mmvq_dot_product(
        const global block_q2_K *data_a, 
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    int sum_d = 0;
    int sum_m = 0;

    const int4 qs_a = repack4(data_a, ib_a, iqs * 4);
    const uchar scale = get_scale(data_a, ib_a, iqs * 4);
    const float2 dm = get_dm(data_a, ib_a);
    const int scale_m = (int)(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.

    sum_d = IMAD(qs_a.x, cache_b_qs[0], sum_d);
    sum_m = IMAD(scale_m, cache_b_qs[0], sum_m);

    sum_d = IMAD(qs_a.y, cache_b_qs[1], sum_d);
    sum_m = IMAD(scale_m, cache_b_qs[1], sum_m);

    sum_d = IMAD(qs_a.z, cache_b_qs[2], sum_d);
    sum_m = IMAD(scale_m, cache_b_qs[2], sum_m);

    sum_d = IMAD(qs_a.w, cache_b_qs[3], sum_d);
    sum_m = IMAD(scale_m, cache_b_qs[3], sum_m);

    sum_d *= (int)(scale & 0xF);

    return cache_b_ds.x * (dm.x * (float)sum_d - dm.y * (float)sum_m);
} 

)";

const char g_codeMmvqImpl_Q3_K[] = R"(
#if 0 // TODO: Revise this
// 2-byte loads for Q3_K blocks (110 bytes)
inline int4 repack4(
        const global block_q3_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q3_K_packed16 *data_a_packed16 = (const global block_q3_K_packed16 *)data_a;

    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

   // bitwise OR to add 4 if hmask is set, subtract later

#define GET_VALS(idx) \
    (as_char2((short)((data_a_packed16[ib_k].qs[qs_idx * 2 + (idx)] >> qs_shift) & (ushort)0x0303)) | \
        as_char2((short)(((data_a_packed16[ib_k].hmask[iqs * 2 + (idx)] >> hm_shift) & (ushort)0x0101) << 2)))

    const char2 vals00 = GET_VALS(0);
    const char2 vals01 = GET_VALS(1);
    const char2 vals10 = GET_VALS(2);
    const char2 vals11 = GET_VALS(3);
    const char2 vals20 = GET_VALS(4);
    const char2 vals21 = GET_VALS(5);
    const char2 vals30 = GET_VALS(6);
    const char2 vals31 = GET_VALS(7);

#undef GET_VALS

    return (int4)(
        as_int((char4)(vals00.x, vals00.y, vals01.x, vals01.y) - (char)4),
        as_int((char4)(vals10.x, vals10.y, vals11.x, vals11.y) - (char)4),
        as_int((char4)(vals20.x, vals20.y, vals21.x, vals21.y) - (char)4),
        as_int((char4)(vals30.x, vals30.y, vals31.x, vals31.y) - (char)4));
}

#else
// 2-byte loads for Q3_K blocks (110 bytes)
inline int4 repack4(
        const global block_q3_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q3_K_packed16 *data_a_packed16 = (const global block_q3_K_packed16 *)data_a;

    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    const uint4 qs = 
        (uint4)( 
            (uint)data_a_packed16[ib_k].qs[qs_idx * 2] |
                ((uint)data_a_packed16[ib_k].qs[qs_idx * 2 + 1] << 16),
            (uint)data_a_packed16[ib_k].qs[qs_idx * 2 + 2] |
                ((uint)data_a_packed16[ib_k].qs[qs_idx * 2 + 3] << 16),
            (uint)data_a_packed16[ib_k].qs[qs_idx * 2 + 4] |
                ((uint)data_a_packed16[ib_k].qs[qs_idx * 2 + 5] << 16),
            (uint)data_a_packed16[ib_k].qs[qs_idx * 2 + 6] |
                ((uint)data_a_packed16[ib_k].qs[qs_idx * 2 + 7] << 16));

    const uint4 hmask = 
        (uint4)( 
            (uint)data_a_packed16[ib_k].hmask[iqs * 2] |
                ((uint)data_a_packed16[ib_k].hmask[iqs * 2 + 1] << 16),
            (uint)data_a_packed16[ib_k].hmask[iqs * 2 + 2] |
                ((uint)data_a_packed16[ib_k].hmask[iqs * 2 + 3] << 16),
            (uint)data_a_packed16[ib_k].hmask[iqs * 2 + 4] |
                ((uint)data_a_packed16[ib_k].hmask[iqs * 2 + 5] << 16),
            (uint)data_a_packed16[ib_k].hmask[iqs * 2 + 6] |
                ((uint)data_a_packed16[ib_k].hmask[iqs * 2 + 7] << 16));

    // bitwise OR to add 4 if hmask is set, subtract later
    const uint vals0 = 
        ((qs.x >> qs_shift) & 0x03030303) | 
            (((hmask.x >> hm_shift) & 0x01010101) << 2);
    const uint vals1 = 
        ((qs.y >> qs_shift) & 0x03030303) |
            (((hmask.y >> hm_shift) & 0x01010101) << 2);
    const uint vals2 = 
        ((qs.z >> qs_shift) & 0x03030303) |
            (((hmask.z >> hm_shift) & 0x01010101) << 2);
    const uint vals3 = 
        ((qs.w >> qs_shift) & 0x03030303) |
            (((hmask.w >> hm_shift) & 0x01010101) << 2);

    return (int4)(
        (int)(((vals0 ^ 0x80808080) - 0x04040404) ^ 0x80808080),
        (int)(((vals1 ^ 0x80808080) - 0x04040404) ^ 0x80808080),
        (int)(((vals2 ^ 0x80808080) - 0x04040404) ^ 0x80808080),
        (int)(((vals3 ^ 0x80808080) - 0x04040404) ^ 0x80808080));
}
#endif

inline float get_d_scale(
        const global block_q3_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 4;

    const char scale = 
        (char)(((data_a[ib_k].scales[is % 8] >> (4 * (is / 8))) & 0x0F0F) |
            (((data_a[ib_k].scales[8 + (is % 4)] >> (2 * (is / 4))) & 0x0303) << 4));
    return (float)data_a[ib_k].d * (float)(scale - 32);
}

inline float mmvq_dot_product(
        const global block_q3_K *data_a, 
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    int q_sum = 0;

    const int4 qs_a = repack4(data_a, ib_a, iqs * 4);
    const float d_scale = get_d_scale(data_a, ib_a, iqs * 4);

    q_sum = IMAD(qs_a.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(qs_a.y, cache_b_qs[1], q_sum);
    q_sum = IMAD(qs_a.z, cache_b_qs[2], q_sum);
    q_sum = IMAD(qs_a.w, cache_b_qs[3], q_sum);

    return cache_b_ds.x * d_scale * (float)q_sum;
} 

)";

const char g_codeMmvqImpl_Q4_K[] = R"(
// 4-byte loads for Q4_K blocks (144 bytes)
inline int4 repack4(
        const global block_q4_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q4_K_packed32 *data_a_packed32 = (const global block_q4_K_packed32 *)data_a;

    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

    const uint vals0 = (data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x0F0F0F0F;
    const uint vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;
    const uint vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F;
    const uint vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F;

    return (int4)(vals0, vals1, vals2, vals3);
}

)";

const char g_codeMmvqImpl_Q5_K[] = R"(
// 4-byte loads for Q5_K blocks (176 bytes)
inline int4 repack4(
        const global block_q5_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q5_K_packed32 *data_a_packed32 = (const global block_q5_K_packed32 *)data_a;

    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

    const uint qh_idx = iqs;
    const uint qh_shift = iqs_k / 8;

    return (int4)(
        ((data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x0F0F0F0F) |
            (((data_a_packed32[ib_k].qh[qh_idx] >> qh_shift) & 0x01010101) << 4),
        ((data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F) |
            (((data_a_packed32[ib_k].qh[qh_idx + 1] >> qh_shift) & 0x01010101) << 4),
        ((data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x0F0F0F0F) |
            (((data_a_packed32[ib_k].qh[qh_idx + 2] >> qh_shift) & 0x01010101) << 4),
        ((data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x0F0F0F0F) |
            (((data_a_packed32[ib_k].qh[qh_idx + 3] >> qh_shift) & 0x01010101) << 4));
}

)";

const char g_codeMmvqImpl_Q45_K[] = R"(
// Common code for Q4_k and Q5_K

float2 get_dm_scale(
        const global A_TYPE *data_a,
        uint ib, 
        uint iqs) {
    const global A_TYPE_PACKED32 *data_a_packed32 = (const global A_TYPE_PACKED32 *)data_a;

    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 8;

    const uint3 scales = 
        (uint3)(
            data_a_packed32[ib_k].scales[0],
            data_a_packed32[ib_k].scales[1],
            data_a_packed32[ib_k].scales[2]);
    const uint scalesoffs = (is & 3) * 8;

    const uint scidx0 = (is < 4) ? 0 : 2;
    const uint scidxshift0 = scalesoffs;
    const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
    const uint mbidx0 = (is < 4) ? 1 : 2;
    const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
    const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

    const uchar sc = (uchar)(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
    const uchar mbyte = (uchar)(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));
    uchar2 scale_dm = (uchar2)(sc, mbyte);

    return convert_float2(data_a_packed32[ib_k].dm) * convert_float2(scale_dm);
}

float mmvq_dot_product(
        const global A_TYPE *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    int q_sum = 0;

    const int4 qs_a = repack4(data_a, ib_a, iqs * 4);
    const float2 dm_scale = get_dm_scale(data_a, ib_a, iqs * 4);

    q_sum = IMAD(qs_a.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(qs_a.y, cache_b_qs[1], q_sum);
    q_sum = IMAD(qs_a.z, cache_b_qs[2], q_sum);
    q_sum = IMAD(qs_a.w, cache_b_qs[3], q_sum);

    return cache_b_ds.x * dm_scale.x * (float)q_sum - dm_scale.y * (cache_b_ds.y / 2);
} 

)";

const char g_codeMmvqImpl_Q6_K[] = R"(
#if 0 // TODO: Revise this
// 2-byte loads for Q6_K blocks (210 bytes)
inline int4 repack4(
        const global block_q6_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q6_K_packed16 *data_a_packed16 = (const global block_q6_K_packed16 *)data_a;

    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

#define GET_VALS(idx) \
    ((as_char2((short)((data_a_packed16[ib_k].ql[ql_idx * 2 + (idx)] >> ql_shift) & (ushort)0x0F0F)) | \
        as_char2((short)(((data_a_packed16[ib_k].qh[qh_idx * 2 + (idx)] >> qh_shift) & (ushort)0x0303) << 4))) - (char)32)

    const char2 vals00 = GET_VALS(0); 
    const char2 vals01 = GET_VALS(1); 
    const char2 vals10 = GET_VALS(2); 
    const char2 vals11 = GET_VALS(3); 
    const char2 vals20 = GET_VALS(4); 
    const char2 vals21 = GET_VALS(5); 
    const char2 vals30 = GET_VALS(6); 
    const char2 vals31 = GET_VALS(7); 

#undef GET_VALS

    return (int4)(
        as_int((char4)(vals00.x, vals00.y, vals01.x, vals01.y)),
        as_int((char4)(vals10.x, vals10.y, vals11.x, vals11.y)),
        as_int((char4)(vals20.x, vals20.y, vals21.x, vals21.y)),
        as_int((char4)(vals30.x, vals30.y, vals31.x, vals31.y)));
} 

#else
// 2-byte loads for Q6_K blocks (210 bytes)
inline int4 repack4(
        const global block_q6_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const global block_q6_K_packed16 *data_a_packed16 = (const global block_q6_K_packed16 *)data_a;

    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;

    const uint qh_idx = (iqs_k / 32) * 8 + iqs;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const uint4 ql = 
        (uint4)( 
            (uint)data_a_packed16[ib_k].ql[ql_idx * 2] |
                ((uint)data_a_packed16[ib_k].ql[ql_idx * 2 + 1] << 16),
            (uint)data_a_packed16[ib_k].ql[ql_idx * 2 + 2] |
                ((uint)data_a_packed16[ib_k].ql[ql_idx * 2 + 3] << 16),
            (uint)data_a_packed16[ib_k].ql[ql_idx * 2 + 4] |
                ((uint)data_a_packed16[ib_k].ql[ql_idx * 2 + 5] << 16),
            (uint)data_a_packed16[ib_k].ql[ql_idx * 2 + 6] |
                ((uint)data_a_packed16[ib_k].ql[ql_idx * 2 + 7] << 16));

    const uint4 qh = 
        (uint4)( 
            (uint)data_a_packed16[ib_k].qh[qh_idx * 2] |
                ((uint)data_a_packed16[ib_k].qh[qh_idx * 2 + 1] << 16),
            (uint)data_a_packed16[ib_k].qh[qh_idx * 2 + 2] |
                ((uint)data_a_packed16[ib_k].qh[qh_idx * 2 + 3] << 16),
            (uint)data_a_packed16[ib_k].qh[qh_idx * 2 + 4] |
                ((uint)data_a_packed16[ib_k].qh[qh_idx * 2 + 5] << 16),
            (uint)data_a_packed16[ib_k].qh[qh_idx * 2 + 6] |
                ((uint)data_a_packed16[ib_k].qh[qh_idx * 2 + 7] << 16));

    const uint vals0 = 
        ((ql.x >> ql_shift) & 0x0F0F0F0F) |
            (((qh.x >> qh_shift) & 0x03030303) << 4);
    const uint vals1 = 
        ((ql.y >> ql_shift) & 0x0F0F0F0F) |
            (((qh.y >> qh_shift) & 0x03030303) << 4);
    const uint vals2 = 
        ((ql.z >> ql_shift) & 0x0F0F0F0F) |
            (((qh.z >> qh_shift) & 0x03030303) << 4);
    const uint vals3 = 
        ((ql.w >> ql_shift) & 0x0F0F0F0F) |
            (((qh.w >> qh_shift) & 0x03030303) << 4);

    return (int4)(
        (int)(((vals0 ^ 0x80808080) - 0x20202020) ^ 0x80808080),
        (int)(((vals1 ^ 0x80808080) - 0x20202020) ^ 0x80808080),
        (int)(((vals2 ^ 0x80808080) - 0x20202020) ^ 0x80808080),
        (int)(((vals3 ^ 0x80808080) - 0x20202020) ^ 0x80808080));
}
#endif

inline float get_d_scale(
        const global block_q6_K *data_a, 
        const uint ib, 
        const uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    return (float)data_a[ib_k].d * (float)data_a[ib_k].scales[iqs_k / 4];
}

inline float mmvq_dot_product(
        const global block_q6_K *data_a, 
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    int q_sum = 0;

    const int4 qs_a = repack4(data_a, ib_a, iqs * 4);
    const float d_scale = get_d_scale(data_a, ib_a, iqs * 4);

    q_sum = IMAD(qs_a.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(qs_a.y, cache_b_qs[1], q_sum);
    q_sum = IMAD(qs_a.z, cache_b_qs[2], q_sum);
    q_sum = IMAD(qs_a.w, cache_b_qs[3], q_sum);

    return cache_b_ds.x * d_scale * (float)q_sum;
} 

)";

const char g_codeMmvqImpl_Mfxp4[] = R"(
const constant char kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
}; 

inline float e8m0_to_fp32(uchar x) {
    uint bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = x;
        bits = bits << 23;
    }

    return as_float(bits);
} 

inline float get_dm(const block_mxfp4 *data_a, const uint ib) {
    return e8m0_to_fp32(data_a[ib].e);
} 

// 1-byte loads for mxfp4 blocks (17 bytes)
inline int2 repack(
        const block_mxfp4 *data_a, 
        const uint ib, 
        const uint iqs) {
    const uint qs = 
        as_uint((uchar4)(
            data_a[ib].qs[iqs * 4],
            data_a[ib].qs[iqs * 4 + 1],
            data_a[ib].qs[iqs * 4 + 2],
            data_a[ib].qs[iqs * 4 + 3]));

    const uchar4 i_a0 = as_uchar4(qs & 0x0F0F0F0F);
    const uchar4 i_a1 = as_uchar4((qs >> 4) & 0x0F0F0F0F);

    return (int2)(
        as_int((char4)(
            kvalues_mxfp4[i_a0.x], 
            kvalues_mxfp4[i_a0.y], 
            kvalues_mxfp4[i_a0.z], 
            kvalues_mxfp4[i_a0.w])),
        as_int((char4)(
            kvalues_mxfp4[i_a1.x], 
            kvalues_mxfp4[i_a1.y], 
            kvalues_mxfp4[i_a1.z], 
            kvalues_mxfp4[i_a1.w])));
}

inline float mul_q8_1(
        const int q_sum, 
        const float da, 
        const float2 dsb, 
        const int sum_divisor) {
    return da * dsb.x * (float)q_sum * 0.5f;
} 

)";

//
//    MulMatQuantVecV2
//

const char g_kernelCodeMmvq[] = R"(
#define B_TYPE block_q8_1_x4

inline void iter(
        const global A_TYPE *data_a,
        const global B_TYPE *data_b,
        const uint ncols,
        const uint batch_stride_b,
        const uint a_offset,
        const uint b_offset,
        float temp[NUM_COLS][NUM_ROWS], 
        const uint first_row, 
        const uint num_rows, 
        const uint tid, 
        const uint i) {
    int cache_b_qs[K_PER_ITER / 4];
    float2 cache_b_ds;

    unroll_for (uint j = 0; j < NUM_COLS; j++) {
        const uint col = i * BLOCK_SIZE + tid * K_PER_ITER;

        // Preload data_b block
        const uint b_block_idx = (j * batch_stride_b + col) / QUANT_K_Q8_1 + b_offset;
        const uint b_qs_idx = tid % (32 / K_PER_ITER);
        const uint b_block_idx_outer = b_block_idx / 4;
        const uint b_block_idx_inner = b_block_idx % 4;

        cache_b_ds = convert_float2(data_b[b_block_idx_outer].ds[b_block_idx_inner]);

#if QUANT_R == 2
        // Assumes K_PER_ITER == 8
        cache_b_qs[0] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + b_qs_idx];
        cache_b_qs[1] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + b_qs_idx + 4];
#else
#if K_PER_ITER == 8
        cache_b_qs[0] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + b_qs_idx * 2];
        cache_b_qs[1] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + b_qs_idx * 2 + 1];
#elif K_PER_ITER == 16
        cache_b_qs[0] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + b_qs_idx * 4];
        cache_b_qs[1] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + b_qs_idx * 4 + 1];
        cache_b_qs[2] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + b_qs_idx * 4 + 2];
        cache_b_qs[3] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + b_qs_idx * 4 + 3];
#elif K_PER_ITER == 32
        cache_b_qs[0] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8];
        cache_b_qs[1] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + 1];
        cache_b_qs[2] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + 2];
        cache_b_qs[3] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + 3];
        cache_b_qs[4] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + 4];
        cache_b_qs[5] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + 5];
        cache_b_qs[6] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + 6];
        cache_b_qs[7] = data_b[b_block_idx_outer].qs[b_block_idx_inner * 8 + 7];
#else
#error unimplemented
#endif
#endif

        uint ibi = first_row * ncols;
        unroll_for (uint n = 0; n < num_rows; n++) {
            const uint a_block_idx = (ibi + col) / QUANT_K_Q8_1 + a_offset;
            ibi += ncols;

            temp[j][n] += 
                mmvq_dot_product(
                    data_a,
                    cache_b_qs,
                    cache_b_ds,
                    a_block_idx, 
                    b_qs_idx);
        }
    }
}

#ifndef MUL_MAT_ID
inline void compute_outputs(
        const global A_TYPE *data_a,
        const global B_TYPE *data_b,
        global float *data_d,
        const uint ncols,
        const uint stride_a,
        const uint stride_b,
        const uint stride_d,
        const uint batch_stride_a,
        const uint batch_stride_b,
        const uint batch_stride_d, 
        const uint base_work_group_y,
        const uint a_dim1,
        const uint b_dim1,
        const uint a_bcast0,
        const uint a_bcast1,
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        local float tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
#endif
        const uint first_row, 
        const uint num_rows) {

    const uint tid = LID_0;

    const uint3 offsets = 
        get_offsets(
            batch_stride_a,
            batch_stride_b,
            batch_stride_d, 
            base_work_group_y,
            a_dim1,
            b_dim1,
            a_bcast0,
            a_bcast1);
    uint a_offset = offsets.x;
    uint b_offset = offsets.y;
    uint d_offset = offsets.z;

    a_offset *= QUANT_K / QUANT_K_Q8_1;
    b_offset /= QUANT_K_Q8_1;

    float temp[NUM_COLS][NUM_ROWS] = {0.0f};

    uint num_iters = ncols / (K_PER_ITER * BLOCK_SIZE);
    if (num_iters * K_PER_ITER * BLOCK_SIZE + K_PER_ITER * tid < ncols) {
        num_iters++;
    }

#define ITER(idx) \
    iter(data_a, data_b, ncols, batch_stride_b, a_offset, b_offset, \
        temp, first_row, num_rows, tid, (idx) * K_PER_ITER)

    int unroll_count = 4;
    uint unrolled_iters = num_iters & ~(unroll_count - 1);

    uint i = 0;
    while (i < unrolled_iters) {
        // Manually partially unroll the loop
        unroll_for (uint k = 0; k < unroll_count; k++) {
            ITER(i);
            i++;
        }
    }

    unroll_count = 2;
    unrolled_iters = num_iters & ~(unroll_count - 1);

    while (i < unrolled_iters) {
        // Manually partially unroll the loop
        unroll_for (uint k = 0; k < unroll_count; k++) {
            ITER(i);
            i++;
        }
    }

    while (i < num_iters) {
        ITER(i);
        i++;
    }

#undef ITER

    reduce_result(
        data_d,
        batch_stride_d,
        temp, 
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        tmpsh,
#endif
        d_offset, 
        first_row, 
        num_rows, 
        tid);
}

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
kernel void mul_mat_vec_q(
        const global A_TYPE *data_a,
        const global B_TYPE *data_b,
        global float *data_d,
        uint ncols,
        uint stride_a,
        uint stride_b,
        uint stride_d,
        uint batch_stride_a,
        uint batch_stride_b,
        uint batch_stride_d, 
        uint base_work_group_y,
        uint a_dim1,
        uint b_dim1,
        uint a_bcast0,
        uint a_bcast1
        SHAPE_INFO_ARGS) {

    data_a += A_BASE / sizeof(A_TYPE);
    data_b += B_BASE / sizeof(B_TYPE);
    data_d += D_BASE;

#ifndef USE_SUBGROUP_ADD_NO_SHMEM
    local float tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

    const uint first_row = NUM_ROWS * (GID_0 + GDIM_0 * GID_2);

    // do NUM_ROWS at a time, unless there aren't enough remaining rows
    if (first_row >= stride_d) {
        return;
    }

    const uint num_rows = min(stride_d - first_row, (uint)NUM_ROWS);
    compute_outputs(
        data_a,
        data_b,
        data_d,
        ncols,
        stride_a,
        stride_b,
        stride_d,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d, 
        base_work_group_y,
        a_dim1,
        b_dim1,
        a_bcast0,
        a_bcast1,
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        tmpsh,
#endif
        first_row, 
        num_rows);
} 

#else
inline void compute_outputs(
        const global A_TYPE *data_a,
        const global B_TYPE *data_b,
        const global int *data_ids,
        global float *data_d,
        const uint ncols,
        const uint stride_a,
        const uint stride_b,
        const uint stride_d,
        const uint batch_stride_a,
        const uint batch_stride_b,
        const uint batch_stride_d, 
        const uint ids_dim3,
        const uint b_dim2,
        const uint ids_i2,
        const uint ids_stride2,
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        local float tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
#endif
        const uint first_row, 
        const uint num_rows) {

    const uint tid = LID_0;

    const uint3 offsets = 
        get_offsets(
            data_ids,
            stride_b,
            stride_d,
            batch_stride_a,
            batch_stride_b,
            batch_stride_d,
            b_dim2,
            ids_i2,
            ids_stride2);
    uint a_offset = offsets.x;
    uint b_offset = offsets.y;
    uint d_offset = offsets.z;

    a_offset *= QUANT_K / QUANT_K_Q8_1;
    b_offset /= QUANT_K_Q8_1;

    float temp[NUM_COLS][NUM_ROWS] = {0.0f};

    uint num_iters = ncols / (K_PER_ITER * BLOCK_SIZE);
    if (num_iters * K_PER_ITER * BLOCK_SIZE + K_PER_ITER * tid < ncols) {
        num_iters++;
    }

#define ITER(idx) \
    iter(data_a, data_b, ncols, batch_stride_b, a_offset, b_offset, \
        temp, first_row, num_rows, tid, (idx) * K_PER_ITER)

    int unroll_count = 4;
    uint unrolled_iters = num_iters & ~(unroll_count - 1);

    uint i = 0;
    while (i < unrolled_iters) {
        // Manually partially unroll the loop
        unroll_for (uint k = 0; k < unroll_count; k++) {
            ITER(i);
            i++;
        }
    }

    unroll_count = 2;
    unrolled_iters = num_iters & ~(unroll_count - 1);

    while (i < unrolled_iters) {
        // Manually partially unroll the loop
        unroll_for (uint k = 0; k < unroll_count; k++) {
            ITER(i);
            i++;
        }
    }

    while (i < num_iters) {
        ITER(i);
        i++;
    }

#undef ITER

    reduce_result(
        data_d,
        batch_stride_d,
        temp, 
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        tmpsh,
#endif
        d_offset, 
        first_row, 
        num_rows, 
        tid);
}

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
kernel void mul_mat_vec_q(
        const global A_TYPE *data_a,
        const global B_TYPE *data_b,
        const global int *data_ids,
        global float *data_d,
        const uint ncols,
        const uint stride_a,
        const uint stride_b,
        const uint stride_d,
        const uint batch_stride_a,
        const uint batch_stride_b,
        const uint batch_stride_d, 
        const uint ids_dim3,
        const uint b_dim2,
        const uint ids_stride2
        SHAPE_INFO_ARGS) {

    data_a += A_BASE / sizeof(A_TYPE);
    data_b += B_BASE / sizeof(B_TYPE);
    data_ids += IDS_BASE;
    data_d += D_BASE;

#ifndef USE_SUBGROUP_ADD_NO_SHMEM
    local float tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

    const uint first_row = NUM_ROWS * GID_0;
    const uint ids_i2 = GID_2;

    // do NUM_ROWS at a time, unless there aren't enough remaining rows
    if (first_row >= stride_d) {
        return;
    }

    const uint num_rows = min(stride_d - first_row, (uint)NUM_ROWS);
    compute_outputs(
        data_a,
        data_b,
        data_ids,
        data_d,
        ncols,
        stride_a,
        stride_b,
        stride_d,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d, 
        ids_dim3,
        b_dim2,
        ids_i2,
        ids_stride2,
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        tmpsh,
#endif
        first_row, 
        num_rows);
} 

#endif

)";

} // namespace

//
//    MulMatQuantVecV2Base
//

const char *MulMatQuantVecV2BaseCode() {
    return g_codeMmvqBase;
}

//
//    MulMatQuantVecV2Defs
//

const char *MulMatQuantVecV2Defs_Q4_0_Code() {
    return g_codeMmvqDefs_Q4_0;
}

const char *MulMatQuantVecV2Defs_Q4_1_Code() {
    return g_codeMmvqDefs_Q4_1;
}

const char *MulMatQuantVecV2Defs_Q5_0_Code() {
    return g_codeMmvqDefs_Q5_0;
}

const char *MulMatQuantVecV2Defs_Q5_1_Code() {
    return g_codeMmvqDefs_Q5_1;
}

const char *MulMatQuantVecV2Defs_Q8_0_Code() {
    return g_codeMmvqDefs_Q8_0;
}

const char *MulMatQuantVecV2Defs_Q8_1_Code() {
    return g_codeMmvqDefs_Q8_1;
}

const char *MulMatQuantVecV2Defs_Q2_K_Code() {
    return g_codeMmvqDefs_Q2_K;
}

const char *MulMatQuantVecV2Defs_Q3_K_Code() {
    return g_codeMmvqDefs_Q3_K;
}

const char *MulMatQuantVecV2Defs_Q4_K_Code() {
    return g_codeMmvqDefs_Q4_K;
}

const char *MulMatQuantVecV2Defs_Q5_K_Code() {
    return g_codeMmvqDefs_Q5_K;
}

const char *MulMatQuantVecV2Defs_Q6_K_Code() {
    return g_codeMmvqDefs_Q6_K;
}

const char *MulMatQuantVecV2Defs_Mxfp4_Code() {
    return g_codeMmvqDefs_Mxfp4;
}

//
//    MulMatQuantVecV2Impl
//

const char *MulMatQuantVecV2Impl_Q4_0_Code() {
    return g_codeMmvqImpl_Q4_0;
}

const char *MulMatQuantVecV2Impl_Q4_1_Code() {
    return g_codeMmvqImpl_Q4_1;
}

const char *MulMatQuantVecV2Impl_Q5_0_Code() {
    return g_codeMmvqImpl_Q5_0;
}

const char *MulMatQuantVecV2Impl_Q5_1_Code() {
    return g_codeMmvqImpl_Q5_1;
}

const char *MulMatQuantVecV2Impl_Q8_0_Code() {
    return g_codeMmvqImpl_Q8_0;
}

const char *MulMatQuantVecV2ImplLegacyCode() {
    return g_codeMmvqImplLegacy;
}

const char *MulMatQuantVecV2Impl_Q2_K_Code() {
    return g_codeMmvqImpl_Q2_K;
}

const char *MulMatQuantVecV2Impl_Q3_K_Code() {
    return g_codeMmvqImpl_Q3_K;
}

const char *MulMatQuantVecV2Impl_Q4_K_Code() {
    return g_codeMmvqImpl_Q4_K;
}

const char *MulMatQuantVecV2Impl_Q5_K_Code() {
    return g_codeMmvqImpl_Q5_K;
}

const char *MulMatQuantVecV2Impl_Q45_K_Code() {
    return g_codeMmvqImpl_Q45_K;
}

const char *MulMatQuantVecV2Impl_Q6_K_Code() {
    return g_codeMmvqImpl_Q6_K;
}

const char *MulMatQuantVecV2Impl_Mxfp4_Code() {
    return g_codeMmvqImpl_Mfxp4;
}

//
//    MulMatQuantVecV2
//

const char *MulMatQuantVecV2KernelCode() {
    return g_kernelCodeMmvq;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

