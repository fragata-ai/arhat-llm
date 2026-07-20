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

// [mul_mv_id_q4_0_f32.cl]
const char g_kernelCodeMulMatId_Q4_0[] = R"(
#define QK4_0 32 

typedef struct {
    half d;
    uchar qs[QK4_0 / 2];
} block_q4_0; 

#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 1  // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16 

inline float block_q_4_0_dot_y(
        const global block_q4_0 *qb_curr,
        float sumy,
        private float *yc,
        int ic) {
    float d = qb_curr->d;
    float2 acc = 0.0f;
    const global ushort *qs = ((const global ushort *)qb_curr + 1 + ic / 2);
    for (int i = 0; i < 8; i += 2) {
        acc.s0 += yc[i + 0] * (qs[i / 2] & 0x000F) + yc[i + 1] * (qs[i / 2] & 0x0F00);
        acc.s1 += yc[i + 8] * (qs[i / 2] & 0x00F0) + yc[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * (-8.0f) + acc.s0 + acc.s1);
} 

__attribute__((intel_reqd_sub_group_size(16))) 
kernel void mul_mat_id_simple_q4_0(
        const global float *src0,
        const global char *src1,
        const global int *src2,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    src2 += SRC2_BASE;
    dst += DST_BASE;

    int r3 = get_group_id(0);
    int r2 = get_group_id(1);
    int i1 = get_group_id(2) / SRC2_D3;
    int i2 = get_group_id(2) % SRC2_D3;

    int k = src2[i1 * SRC2_S2 + i2 * SRC2_S3];

    const global float *src0_cur = src0 + i1 * SRC0_S1 + (i2 % SRC0_D2) * SRC0_S2;
    const global char *src1_cur = src1 + k * SRC1_S1;

    global float *dst_cur = dst + i1 * DST_S1 + i2 * DST_S2;
 
    int nb = SRC1_D3 / QK4_0;
    int sb = SRC1_S2 / (2 + QK4_0 / 2); // stride in block units

    int first_row = (r3 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    ulong offset_src1 = first_row * SRC1_S2;

    const global float *y = src0_cur + r2 * SRC0_S2;
    const global block_q4_0 *x = (const global block_q4_0 *)(src1_cur + offset_src1);

    float yc[16];
    float sumf[N_DST] = {0.0f};

    int ix = get_sub_group_local_id() / 2;
    int ic = 8 * (get_sub_group_local_id() % 2);

    const global float *yb = y + ix * QK4_0 + ic;

    // each thread in a SIMD group deals with half a block
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH / 2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i + 1];
            yc[i + 0] = yb[i + 0];
            yc[i + 1] = yb[i + 1] / 256.0f;
            sumy += yb[i + 16] + yb[i + 17];
            yc[i + 8] = yb[i + 16] / 16.0f;
            yc[i + 9] = yb[i + 17] / 4096.0f;
        }

        for (int row = 0; row < N_DST; row++) {
            sumf[row] += block_q_4_0_dot_y(x + ib + row * sb, sumy, yc, ic);
        }

        yb += QK4_0 * (N_SIMDWIDTH / 2);
    }

    global float *dst_ptr = dst_cur + r2 * DST_S2;

    for (int row = 0; row < N_DST; row++) {
        float tot = sub_group_reduce_add(sumf[row]);
        if (first_row + row < SRC1_D2 && get_sub_group_local_id() == 0) {
            dst_ptr[first_row + row] = tot;
        }
    }
}

)";

// [mul_mv_id_q4_k_f32.cl]
const char g_kernelCodeMulMatId_Q4_K[] = R"(
#define QK_K 256
#define K_SCALE_SIZE 12

// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
typedef struct {
    half d;                     // super-block scale for quantized scales
    half dmin;                  // super-block scale for quantized mins
    uchar scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uchar qs[QK_K / 2];         // 4-bit quants
} block_q4_K;

#define N_DST 4        // number of rows each SIMD group works on
#define N_SIMDGROUP 1  // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // SIMD group size

// number of (super) blocks each subgroup processes
// each thread in a subgroup processes a block (32 weights)
#define BLOCK_STRIDE (N_SIMDWIDTH / 8)
 
__attribute__((intel_reqd_sub_group_size(16))) 
kernel void mul_mat_id_simple_q4_k(
        const global float *src0,
        const global char *src1,
        const global int *src2,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    src2 += SRC2_BASE;
    dst += DST_BASE;

    int r3 = get_group_id(0);
    int r2 = get_group_id(1);
    int i1 = get_group_id(2) / SRC2_D3;
    int i2 = get_group_id(2) % SRC2_D3;

    int k = src2[i1 * SRC2_S2 + i2 * SRC2_S3];

    const global float *src0_cur = src0 + i1 * SRC0_S1 + (i2 % SRC0_D2) * SRC0_S2;
    const global char *src1_cur = src1 + k * SRC1_S1;

    global float *dst_cur = dst + i1 * DST_S1 + i2 * DST_S2;
 
    int nb = SRC1_D3 / QK_K;

    int first_row = (r3 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    ulong offset_src0 = r2 * SRC0_S2;
    ulong offset_src1 = first_row * SRC1_S2;

    ushort kmask1 = 0x3f3f;
    ushort kmask2 = 0x0f0f;
    ushort kmask3 = 0xc0c0;

    int ix = get_sub_group_local_id() / 8; // super block index
    int it = get_sub_group_local_id() % 8; // block index (inside super block)
    int iq = it / 4;                       // 0 or 1 - first or second half of the super block
    int ir = it % 4;                       // 0...3 - block index in the half super block

    const global float *y = src0_cur + offset_src0;
    const global block_q4_K *x = (const global block_q4_K *)(src1_cur + offset_src1);

    float yl[16];
    float yh[16];
    float sumf[N_DST] = {0.0f};

    const global float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    ushort sc16[4];
    uchar *sc8 = (uchar *)sc16; 

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 8; i++) {
            yl[i + 0] = y4[i + 0];
            sumy.s0 += yl[i + 0];

            yl[i + 8] = y4[i + 32];
            sumy.s1 += yl[i + 8];

            yh[i + 0] = y4[i + 128];
            sumy.s2 += yh[i + 0];

            yh[i + 8] = y4[i + 160];
            sumy.s3 += yh[i + 8];
        }

        const global ushort *sc = (global ushort *)x[ib].scales + iq;
        const global ushort *q1 = (global ushort *)x[ib].qs + 16 * iq + 4 * ir;
        const global half *dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            const global ushort *q2 = q1 + 32;

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int i = 0; i < 8; i += 2) {
                acc1.s0 += yl[i + 0] * (q1[i / 2] & 0x000F);
                acc1.s1 += yl[i + 1] * (q1[i / 2] & 0x0F00);
                acc1.s2 += yl[i + 8] * (q1[i / 2] & 0x00F0);
                acc1.s3 += yl[i + 9] * (q1[i / 2] & 0xF000);
                acc2.s0 += yh[i + 0] * (q2[i / 2] & 0x000F);
                acc2.s1 += yh[i + 1] * (q2[i / 2] & 0x0F00);
                acc2.s2 += yh[i + 8] * (q2[i / 2] & 0x00F0);
                acc2.s3 += yh[i + 9] * (q2[i / 2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += 
                dall * ((acc1.s0 + 1.0f / 256.0f * acc1.s1) * sc8[0] +
                    (acc1.s2 + 1.0f / 256.0f * acc1.s3) * sc8[1] * 1.0f / 16.0f +
                    (acc2.s0 + 1.0f / 256.0f * acc2.s1) * sc8[4] +
                    (acc2.s2 + 1.0f / 256.0f * acc2.s3) * sc8[5] * 1.0f / 16.0f) -
                dmin * (sumy.s0 * sc8[2] + sumy.s1 * sc8[3] + sumy.s2 * sc8[6] + sumy.s3 * sc8[7]);

            q1 += SRC1_S2 / 2;
            sc += SRC1_S2 / 2;
            dh += SRC1_S2 / 2;
        }

        y4 += BLOCK_STRIDE * QK_K;
    }


    global float *dst_ptr = dst_cur + r2 * DST_S2;

    for (int row = 0; row < N_DST; row++) {
        float tot = sub_group_reduce_add(sumf[row]);
        if (first_row + row < SRC1_D2) {
            if (get_sub_group_local_id() == 0) {
                dst_ptr[first_row + row] = tot;
            }
        }
    } 
}

)";

// [mul_mv_id_q5_0_f32.cl]
const char g_kernelCodeMulMatId_Q5_0[] = R"(
#define QK5_0 32 

typedef struct {
    half d;
    uchar qh[4];
    uchar qs[QK5_0 / 2];
} block_q5_0; 

#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 1  // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16 

inline float block_q_5_0_dot_y(
        const global block_q5_0 *qb_curr,
        float sumy,
        private float *yc,
        int ic) {
    float d = qb_curr->d;
    const global ushort *qh = (const global ushort *)qb_curr->qh;
    uint h0 = qh[0] >> ic;
    uint h1 = qh[1] >> ic;
    const global ushort *qs = (const global ushort *)(qb_curr->qs + ic);
    float2 acc = 0.0f;
    for (int i = 0; i < 8; i += 2) {
        uint xs = qs[i / 2];
        uint h00 = ((h0 >> (i + 0)) << 4) & 0x10;
        uint h01 = ((h0 >> (i + 1)) << 12) & 0x1000;
        uint h10 = ((h1 >> (i + 0)) << 8) & 0x100;
        uint h11 = ((h1 >> (i + 1)) << 16) & 0x10000;
        acc.s0 += yc[i + 0] * ((xs & 0x000F) | h00) + yc[i + 1] * ((xs & 0x0F00) | h01);
        acc.s1 += yc[i + 8] * ((xs & 0x00F0) | h10) + yc[i + 9] * ((xs & 0xF000) | h11);
    }
    return d * (sumy * (-16.0f) + acc.s0 + acc.s1);
} 

__attribute__((intel_reqd_sub_group_size(16))) 
kernel void mul_mat_id_simple_q5_0(
        const global float *src0,
        const global char *src1,
        const global int *src2,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    src2 += SRC2_BASE;
    dst += DST_BASE;

    int r3 = get_group_id(0);
    int r2 = get_group_id(1);
    int i1 = get_group_id(2) / SRC2_D3;
    int i2 = get_group_id(2) % SRC2_D3;

    int k = src2[i1 * SRC2_S2 + i2 * SRC2_S3];

    const global float *src0_cur = src0 + i1 * SRC0_S1 + (i2 % SRC0_D2) * SRC0_S2;
    const global char *src1_cur = src1 + k * SRC1_S1;

    global float *dst_cur = dst + i1 * DST_S1 + i2 * DST_S2;

    int nb = SRC1_D3 / QK5_0;
    int sb = SRC1_S2 / (6 + QK5_0 / 2); // stride in block units

    int first_row = (r3 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    ulong offset_src1 = first_row * SRC1_S2;

    const global float *y = src0_cur + r2 * SRC0_S2;
    const global block_q5_0 *x = (const global block_q5_0 *)(src1_cur + offset_src1);

    float yc[16];
    float sumf[N_DST] = {0.0f};

    int ix = get_sub_group_local_id() / 2;
    int ic = 8 * (get_sub_group_local_id() % 2);

    const global float *yb = y + ix * QK5_0 + ic;

    // each thread in a SIMD group deals with half a block
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH / 2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i + 1];
            yc[i + 0] = yb[i + 0];
            yc[i + 1] = yb[i + 1] / 256.0f;
            sumy += yb[i + 16] + yb[i + 17];
            yc[i + 8] = yb[i + 16] / 16.0f;
            yc[i + 9] = yb[i + 17] / 4096.0f;
        }

        for (int row = 0; row < N_DST; row++) {
            sumf[row] += block_q_5_0_dot_y(x + ib + row * sb, sumy, yc, ic);
        }

        yb += QK5_0 * (N_SIMDWIDTH / 2);
    }

    global float *dst_ptr = dst_cur + r2 * DST_S2;

    for (int row = 0; row < N_DST; row++) {
        float tot = sub_group_reduce_add(sumf[row]);
        if (first_row + row < SRC1_D2 && get_sub_group_local_id() == 0) {
            dst_ptr[first_row + row] = tot;
        }
    }
}

)";

// [mul_mv_id_q6_k_f32.cl]
const char g_kernelCodeMulMatId_Q6_K[] = R"(
#define QK_K 256 

typedef struct {
    uchar ql[QK_K / 2];      // quants, lower 4 bits
    uchar qh[QK_K / 4];      // quants, upper 2 bits
    char  scales[QK_K / 16]; // scales, quantized with 8 bits
    half d;                  // super-block scale
} block_q6_K; 

#define N_DST 1        // number of rows each SIMD group works on
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // SIMD group size 

#define BLOCK_STRIDE (N_SIMDWIDTH / 16) // number of blocks each subgroup processes 

__attribute__((intel_reqd_sub_group_size(16))) 
kernel void mul_mat_id_simple_q6_k(
        const global float *src0,
        const global char *src1,
        const global int *src2,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    src2 += SRC2_BASE;
    dst += DST_BASE;

    int r3 = get_group_id(0);
    int r2 = get_group_id(1);
    int i1 = get_group_id(2) / SRC2_D3;
    int i2 = get_group_id(2) % SRC2_D3;

    int k = src2[i1 * SRC2_S2 + i2 * SRC2_S3];

    const global float *src0_cur = src0 + i1 * SRC0_S1 + (i2 % SRC0_D2) * SRC0_S2;
    const global char *src1_cur = src1 + k * SRC1_S1;

    global float *dst_cur = dst + i1 * DST_S1 + i2 * DST_S2;
 
    int nb = SRC1_D3 / QK_K;

    int row = r3 * N_SIMDGROUP + get_sub_group_id();

    if (row >= SRC1_D2) {
        return;
    }

    const global float *yy = src0_cur + r2 * SRC0_S2;
    const global block_q6_K *x = (global block_q6_K *)(src1_cur + row * SRC1_S2);

    uchar kmask1 = 0x03;
    uchar kmask2 = 0x0C;
    uchar kmask3 = 0x30;
    uchar kmask4 = 0xC0;

    float sumf = 0.0f;

    int tid = get_sub_group_local_id() / BLOCK_STRIDE;
    int ix = get_sub_group_local_id() % BLOCK_STRIDE;
    int ip = tid / 8;          // first or second half of (super) block (0 or 1) 
    int ic = tid % 8;          // each half has 8 parts, one per scale 
    int n = 4;                 // 4 scales at a time (and 4 sums) 
    int c0 = n * ic;           // offset into half-block
    int is = 8 * ip + c0 / 16;

    int y_offset = 128 * ip + c0;
    int q_offset_l = 64 * ip + c0;
    int q_offset_h = 32 * ip + c0;

    for (int i = ix; i < nb; i += BLOCK_STRIDE) {
        const global uchar *q1 = x[i].ql + q_offset_l;
        const global uchar *q2 = q1 + QK_K / 8;
        const global uchar *qh = x[i].qh + q_offset_h;
        const global char *sc = x[i].scales + is;

        const global float *y = yy + i * QK_K + y_offset;

        float dall = x[i].d;

        float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};

        sums.s0 += y[0 + 0] * ((float)((q1[0] & 0xF) | ((qh[0] & kmask1) << 4)) - 32.0f);
        sums.s1 += y[0 + 32] * ((float)((q2[0] & 0xF) | ((qh[0] & kmask2) << 2)) - 32.0f);
        sums.s2 += y[0 + 64] * ((float)((q1[0] >> 4) | ((qh[0] & kmask3) << 0)) - 32.0f);
        sums.s3 += y[0 + 96] * ((float)((q2[0] >> 4) | ((qh[0] & kmask4) >> 2)) - 32.0f);

        sums.s0 += y[1 + 0] * ((float)((q1[1] & 0xF) | ((qh[1] & kmask1) << 4)) - 32.0f);
        sums.s1 += y[1 + 32] * ((float)((q2[1] & 0xF) | ((qh[1] & kmask2) << 2)) - 32.0f);
        sums.s2 += y[1 + 64] * ((float)((q1[1] >> 4) | ((qh[1] & kmask3) << 0)) - 32.0f);
        sums.s3 += y[1 + 96] * ((float)((q2[1] >> 4) | ((qh[1] & kmask4) >> 2)) - 32.0f);

        sums.s0 += y[2 + 0] * ((float)((q1[2] & 0xF) | ((qh[2] & kmask1) << 4)) - 32.0f);
        sums.s1 += y[2 + 32] * ((float)((q2[2] & 0xF) | ((qh[2] & kmask2) << 2)) - 32.0f);
        sums.s2 += y[2 + 64] * ((float)((q1[2] >> 4) | ((qh[2] & kmask3) << 0)) - 32.0f);
        sums.s3 += y[2 + 96] * ((float)((q2[2] >> 4) | ((qh[2] & kmask4) >> 2)) - 32.0f);

        sums.s0 += y[3 + 0] * ((float)((q1[3] & 0xF) | ((qh[3] & kmask1) << 4)) - 32.0f);
        sums.s1 += y[3 + 32] * ((float)((q2[3] & 0xF) | ((qh[3] & kmask2) << 2)) - 32.0f);
        sums.s2 += y[3 + 64] * ((float)((q1[3] >> 4) | ((qh[3] & kmask3) << 0)) - 32.0f);
        sums.s3 += y[3 + 96] * ((float)((q2[3] >> 4) | ((qh[3] & kmask4) >> 2)) - 32.0f);

        sumf += dall * (sums.s0 * sc[0] + sums.s1 * sc[2] + sums.s2 * sc[4] + sums.s3 * sc[6]);
    }

    global float *dst_ptr = dst_cur + r2 * DST_S2;

    float tot = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        dst_ptr[row] = tot;
    }
}

)";

// [mul_mv_id_q8_0_f32.cl]
const char g_kernelCodeMulMatId_Q8_0[] = R"(
#define QK8_0 32

typedef struct {
    half d;         // delta
    char qs[QK8_0]; // quants
} block_q8_0;

#define NB_Q8_0 8

#define N_R0_Q8_0 4    // number of rows each subgroup works on
#define N_SG_Q8_0 2    // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size 

__attribute__((intel_reqd_sub_group_size(16))) 
kernel void mul_mat_id_simple_q8_0(
        const global float *src0,
        const global char *src1,
        const global int *src2,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    src2 += SRC2_BASE;
    dst += DST_BASE;

    int r3 = get_group_id(0);
    int r2 = get_group_id(1);
    int i1 = get_group_id(2) / SRC2_D3;
    int i2 = get_group_id(2) % SRC2_D3;

    int k = src2[i1 * SRC2_S2 + i2 * SRC2_S3];

    const global float *src0_cur = src0 + i1 * SRC0_S1 + (i2 % SRC0_D2) * SRC0_S2;
    const global char *src1_cur = src1 + k * SRC1_S1;

    global float *dst_cur = dst + i1 * DST_S1 + i2 * DST_S2;

    int nb = SRC1_D3 / QK8_0;

    int first_row = (r3 * N_SG_Q8_0 + get_sub_group_id()) * N_R0_Q8_0;

    ulong offset_src0 = r2 * SRC0_S2;
    const global float *y = src0_cur + offset_src0;

    // pointers to src0 rows
    const global block_q8_0 *ax[N_R0_Q8_0];
    for (int row = 0; row < N_R0_Q8_0; row++) {
        ulong offset_src1 = (first_row + row) * SRC1_S2;
        ax[row] = (const global block_q8_0 *)(src1_cur + offset_src1);
    }

    float yc[NB_Q8_0];
    float sumf[N_R0_Q8_0] = {0.0f};

    short ix = get_sub_group_local_id() / 4;
    short ic = get_sub_group_local_id() % 4;

    const global float *yb = y + ix * QK8_0 + ic * NB_Q8_0;

    // each thread handles NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH / 4) {
        for (short i = 0; i < NB_Q8_0; i++) {
            yc[i] = yb[i];
        }

        for (short row = 0; row < N_R0_Q8_0; row++) {
            const global char *qs = ax[row][ib].qs + ic * NB_Q8_0;
            float sumq = 0.0f;
            for (short iq = 0; iq < NB_Q8_0; iq++) {
                sumq += qs[iq] * yc[iq];
            }
            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += N_SIMDWIDTH * NB_Q8_0;
    }

    global float *dst_ptr = dst_cur + r2 * DST_S2;

    for (int row = 0; row < N_R0_Q8_0; row++) {
        float tot = sub_group_reduce_add(sumf[row]);

        if (get_sub_group_local_id() == 0 && first_row + row < SRC1_D2) {
            dst_ptr[first_row + row] = tot;
        }
    } 
}

)";

// [mul_mv_id_mxfp4_f32.cl]
const char g_kernelCodeMulMatId_Mxfp4[] = R"(
#define QK_MXFP4 32

typedef struct {
    uchar e; // E8M0
    uchar qs[QK_MXFP4 / 2];
} block_mxfp4;

constant static float kvalues_mxfp4_f[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float e8m0_to_fp32(uchar x) {
    int bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint) x << 23;
    }
    return as_float(bits);
}

#define N_R0_MXFP4 2   // number of rows each subgroup works on
#define N_SG_MXFP4 2   // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size 

__attribute__((intel_reqd_sub_group_size(16)))
kernel void mul_mat_id_simple_mxfp4(
        const global float *src0,
        const global char *src1,
        const global int *src2,
        global float *dst) {
    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    src2 += SRC2_BASE;
    dst += DST_BASE;

    int r3 = get_group_id(0);
    int r2 = get_group_id(1);
    int i1 = get_group_id(2) / SRC2_D3;
    int i2 = get_group_id(2) % SRC2_D3;

    int k = src2[i1 * SRC2_S2 + i2 * SRC2_S3];

    const global float *src0_cur = src0 + i1 * SRC0_S1 + (i2 % SRC0_D2) * SRC0_S2;
    const global char *src1_cur = src1 + k * SRC1_S1;

    global float *dst_cur = dst + i1 * DST_S1 + i2 * DST_S2;
 
    int nb = SRC1_D3 / QK_MXFP4;
    int sb = SRC1_S2 / (1 + QK_MXFP4 / 2); // stride in block units

    int first_row = (r3 * N_SG_MXFP4 + get_sub_group_id()) * N_R0_MXFP4;

    ulong offset_src0 = r2 * SRC0_S2;
    ulong offset_src1 = first_row * SRC1_S2;

    const global float *y = src0_cur + offset_src0;
    const global block_mxfp4 *x = (global block_mxfp4 *)(src1_cur + offset_src1);

    short ix = get_sub_group_local_id() / 2; // 0...15
    short it = get_sub_group_local_id() % 2; // 0 or 1

    local float values[N_SIMDWIDTH];

    values[get_sub_group_local_id()] = kvalues_mxfp4_f[get_sub_group_local_id() % 16];

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 yc[4];
    float sumf[N_R0_MXFP4] = {0.0f};

    const global float *yb = y + ix * QK_MXFP4 + it * 8;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        const global float4 *y4 = (const global float4 *)yb;
        yc[0] = y4[0];
        yc[1] = y4[4];
        yc[2] = y4[1];
        yc[3] = y4[5];

        for (short row = 0; row < N_R0_MXFP4; row++) {
            const global block_mxfp4 *xb = x + row * sb + ib;
            const global uchar *q2 = xb->qs + 8 * it;

            float4 acc1 = 
                yc[0] *
                    (float4)(
                        values[q2[0] & 0x0F], 
                        values[q2[1] & 0x0F], 
                        values[q2[2] & 0x0F], 
                        values[q2[3] & 0x0F]);
            float4 acc2 = 
                yc[1] *
                    (float4)(
                        values[q2[0] >> 4], 
                        values[q2[1] >> 4], 
                        values[q2[2] >> 4], 
                        values[q2[3] >> 4]);
            float4 acc3 = 
                yc[2] * 
                    (float4)(
                        values[q2[4] & 0x0F], 
                        values[q2[5] & 0x0F], 
                        values[q2[6] & 0x0F], 
                        values[q2[7] & 0x0F]);
            float4 acc4 = 
                yc[3] * 
                    (float4)(
                        values[q2[4] >> 4], 
                        values[q2[5] >> 4], 
                        values[q2[6] >> 4], 
                        values[q2[7] >> 4]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(xb->e) * ((acc1.s0 + acc1.s1) + (acc1.s2 + acc1.s3));
        }

        yb += (N_SIMDWIDTH / 2) * QK_MXFP4;
    }

    global float *dst_ptr = dst_cur + r2 * DST_S2;

    for (int row = 0; row < N_R0_MXFP4 && first_row + row < DST_D3; row++) {
        float sum_all = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0) {
            dst_ptr[first_row + row] = sum_all;
        }
    } 
}

)";

} // namespace

const char *MulMatIdQuantSimple_Q4_0_KernelCode() {
    return g_kernelCodeMulMatId_Q4_0;
}

const char *MulMatIdQuantSimple_Q4_K_KernelCode() {
    return g_kernelCodeMulMatId_Q4_K;
}

const char *MulMatIdQuantSimple_Q5_0_KernelCode() {
    return g_kernelCodeMulMatId_Q5_0;
}

const char *MulMatIdQuantSimple_Q6_K_KernelCode() {
    return g_kernelCodeMulMatId_Q6_K;
}

const char *MulMatIdQuantSimple_Q8_0_KernelCode() {
    return g_kernelCodeMulMatId_Q8_0;
}

const char *MulMatIdQuantSimple_Mxfp4_KernelCode() {
    return g_kernelCodeMulMatId_Mxfp4;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

