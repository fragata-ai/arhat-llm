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

const char g_codeVecDotQuantCommon[] = R"(
#define QK_K 256
#define K_SCALE_SIZE 12

#define QK8_1 32

typedef struct {
    half2 ds;       // [delta, d * sum(qs[*])]
    char qs[QK8_1]; // quants
} block_q8_1; 

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1 

inline int get_int_b1(const void *x, const int i32) {
    const uchar *x8 = (const uchar *)x + 4 * i32;
    return as_int((uchar4)(x8[0], x8[1], x8[2], x8[3]));
}

inline int get_int_b2(const void *x, const int i32) {
    // assume at least 2 byte alignment
    const ushort *x16 = (const ushort *)x + 2 * i32; 
    return as_int((ushort2)(x16[0], x16[1]));
}

inline int get_int_b4(const void *x, const int i32) {
    // assume at least 4 byte alignment
    return ((const int *)x)[i32]; 
} 

// q4 contains 8 indices with 4 bit each.
// This function selects those bytes from table that are at those indices and returns them as int2.
// The first int contains the bytes with even indices in q4, the second int contains the bytes
// with odd indices in q4.
inline int2 get_int_from_table_16(const int q4, const constant char *table) { 
    // generic implementation
    const int q0_32 = (q4 >> 0) & 0x0F0F0F0F;
    const char4 q0_8 = as_char4(q0_32);
    const char4 val0_8 = (char4)(table[q0_8[0]], table[q0_8[1]], table[q0_8[2]], table[q0_8[3]]);

    const int q1_32 = (q4 >> 4) & 0x0F0F0F0F;
    const char4 q1_8 = as_char4(q1_32);
    const char4 val1_8 = (char4)(table[q1_8[0]], table[q1_8[1]], table[q1_8[2]], table[q1_8[3]]);

    return (int2)(as_int(val0_8), as_int(val1_8));
}

inline float e8m0_to_fp32(uchar x) {
    uint bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint)x << 23;
    }
    return as_float(bits);
}

#define IMAD(a, b, c) imad(as_char4(a), as_char4(b), c)
#define ISUB_SAT(a, b) as_int(sub_sat(as_char4(a), as_char4(b)))

)";

//
//    VecDotQuantDefs
//

const char g_codeVecDotDefs_Q4_0[] = R"(
#define QK4_0 32

typedef struct {
    half d;              // delta
    uchar qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0; 

#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2 

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ 4

)";

const char g_codeVecDotDefs_Q4_1[] = R"(
#define QK4_1 32

typedef struct {
    half2 dm;
    uchar qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1; 

#define QI4_1 (QK4_1 / (4 * QR4_1))
#define QR4_1 2 

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ 4

)";

const char g_codeVecDotDefs_Q5_0[] = R"(
#define QK5_0 32

typedef struct {
    half d;              // delta
    uchar qh[4];         // 5-th bit of quants
    uchar qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0; 

#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QR5_0 2

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ 4

)";

const char g_codeVecDotDefs_Q5_1[] = R"(
#define QK5_1 32

typedef struct {
    half2 dm;
    uchar qh[4];         // 5-th bit of quants
    uchar qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1; 

#define QI5_1 (QK5_1 / (4 * QR5_1))
#define QR5_1 2

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ 4

)";

const char g_codeVecDotDefs_Q8_0[] = R"(
#define QK8_0 32

typedef struct {
    half d;         // delta
    char qs[QK8_0]; // quants
} block_q8_0; 

#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

)";

const char g_codeVecDotDefs_Q2_K[] = R"(
// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight

typedef struct {
    uchar scales[QK_K / 16]; // scales and mins, quantized with 4 bits
    uchar qs[QK_K / 4];      // quants
    half2 dm;                // super-block scales for quantized scales and mins
} block_q2_K; 

#define QI2_K (QK_K / (4 * QR2_K))
#define QR2_K 4

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ 4

)";

const char g_codeVecDotDefs_Q3_K[] = R"(
// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight

typedef struct {
    uchar hmask[QK_K / 8]; // quants - high bit
    uchar qs[QK_K / 4];    // quants - low 2 bits
    uchar scales[12];      // scales, quantized with 6 bits
    half d;                // super-block scale
} block_q3_K; 

#define QI3_K (QK_K / (4 * QR3_K))
#define QR3_K 4

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ 2

)";

const char g_codeVecDotDefs_Q4_K[] = R"(
// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight

typedef struct {
    half2 dm;                   // super-block scales for quantized scales and mins
    uchar scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uchar qs[QK_K / 2];         // 4--bit quants
} block_q4_K; 

#define QI4_K (QK_K / (4 * QR4_K))
#define QR4_K 2

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ 8

)";

const char g_codeVecDotDefs_Q5_K[] = R"(
// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight

typedef struct {
    half2 dm;                   // super-block scales for quantized scales and mins
    uchar scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uchar qh[QK_K / 8];         // quants, high bit
    uchar qs[QK_K / 2];         // quants, low 4 bits
} block_q5_K; 

#define QI5_K (QK_K / (4 * QR5_K))
#define QR5_K 2

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ 8

)";

const char g_codeVecDotDefs_Q6_K[] = R"(
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight

typedef struct {
    uchar ql[QK_K / 2];      // quants, lower 4 bits
    uchar qh[QK_K / 4];      // quants, upper 2 bits
    char scales[QK_K / 16];  // scales, quantized with 8 bits
    half d;                  // super-block scale
} block_q6_K; 

#define QI6_K (QK_K / (4 * QR6_K))
#define QR6_K 2

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ 8

)";

const char g_codeVecDotDefs_Mxfp4[] = R"(
// e2m1 values (doubled)
// ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
constant char kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

#define QK_MXFP4 32

typedef struct {
    uchar e; // E8M0
    uchar qs[QK_MXFP4 / 2];
} block_mxfp4; 

#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))
#define QR_MXFP4 2 

#define VDR_MXFP4_Q8_1_MMVQ 2
#define VDR_MXFP4_Q8_1_MMQ 4

)";

//
//    VecDot
//

const char g_codeVecDot_Q4_0[] = R"(
#if !USE_SOA_Y
inline float vec_dot_q4_0_q8_1(
        const global void *vbq, 
        const global block_q8_1 *bq8_1, 
        const int kbx, 
        const int iqs) {

    const global block_q4_0 *bq4_0 = (const global block_q4_0 *)vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2 * VDR_Q4_0_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; i++) {
        v[i] = get_int_b2(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl(v, u, bq4_0->d, bq8_1->ds);
}

#else
inline float vec_dot_q4_0_q8_1(
        const global void *vbq, 
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
        const int kbx, 
        const int iqs) {

    const global block_q4_0 *bq4_0 = (const global block_q4_0 *)vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2 * VDR_Q4_0_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; i++) {
        v[i] = get_int_b2(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_b4(q8_1_qs, iqs + i);
        u[2 * i + 1] = get_int_b4(q8_1_qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl(v, u, bq4_0->d, *q8_1_ds);
}
#endif

)";

const char g_codeVecDot_Q4_1[] = R"(
#if !USE_SOA_Y
inline float vec_dot_q4_1_q8_1(
        const global void *vbq, 
        const global block_q8_1 *bq8_1, 
        const int kbx, 
        const int iqs) {

    const global block_q4_1 *bq4_1 = (const global block_q4_1 *)vbq + kbx;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2 * VDR_Q4_1_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; i++) {
        v[i] = get_int_b4(bq4_1->qs, iqs + i);
        u[2 * i + 0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl(v, u, bq4_1->dm, bq8_1->ds);
} 

#else
inline float vec_dot_q4_1_q8_1(
        const global void *vbq, 
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
        const int kbx, 
        const int iqs) {

    const global block_q4_1 *bq4_1 = (const global block_q4_1 *)vbq + kbx;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2 * VDR_Q4_1_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; i++) {
        v[i] = get_int_b4(bq4_1->qs, iqs + i);
        u[2 * i + 0] = get_int_b4(q8_1_qs, iqs + i);
        u[2 * i + 1] = get_int_b4(q8_1_qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl(v, u, bq4_1->dm, *q8_1_ds);
} 
#endif

)";

const char g_codeVecDot_Q5_0[] = R"(
#if !USE_SOA_Y
inline float vec_dot_q5_0_q8_1(
        const global void *vbq, 
        const global block_q8_1 *bq8_1, 
        const int kbx, 
        const int iqs) {

    const global block_q5_0 *bq5_0 = (const global block_q5_0 *)vbq + kbx;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int u[2 * VDR_Q5_0_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; i++) {
        vl[i] = get_int_b2(bq5_0->qs, iqs + i);
        vh[i] = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2 * i + 0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl(vl, vh, u, bq5_0->d, bq8_1->ds);
} 

#else
inline float vec_dot_q5_0_q8_1(
        const global void *vbq, 
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
        const int kbx, 
        const int iqs) {

    const global block_q5_0 *bq5_0 = (const global block_q5_0 *)vbq + kbx;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int u[2 * VDR_Q5_0_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; i++) {
        vl[i] = get_int_b2(bq5_0->qs, iqs + i);
        vh[i] = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2 * i + 0] = get_int_b4(q8_1_qs, iqs + i);
        u[2 * i + 1] = get_int_b4(q8_1_qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl(vl, vh, u, bq5_0->d, *q8_1_ds);
} 
#endif

)";

const char g_codeVecDot_Q5_1[] = R"(
#if !USE_SOA_Y
inline float vec_dot_q5_1_q8_1(
        const global void *vbq, 
        const global block_q8_1 *bq8_1, 
        const int kbx, 
        const int iqs) {

    const global block_q5_1 *bq5_1 = (const global block_q5_1 *)vbq + kbx;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int u[2 * VDR_Q5_1_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; i++) {
        vl[i] = get_int_b4(bq5_1->qs, iqs + i);
        vh[i] = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2 * i + 0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl(vl, vh, u, bq5_1->dm, bq8_1->ds);
} 

#else
inline float vec_dot_q5_1_q8_1(
        const global void *vbq, 
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
        const int kbx, 
        const int iqs) {

    const global block_q5_1 *bq5_1 = (const global block_q5_1 *)vbq + kbx;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int u[2 * VDR_Q5_1_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; i++) {
        vl[i] = get_int_b4(bq5_1->qs, iqs + i);
        vh[i] = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2 * i + 0] = get_int_b4(q8_1_qs, iqs + i);
        u[2 * i + 1] = get_int_b4(q8_1_qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl(vl, vh, u, bq5_1->dm, *q8_1_ds);
} 
#endif

)";

const char g_codeVecDot_Q8_0[] = R"(
#if !USE_SOA_Y
inline float vec_dot_q8_0_q8_1(
        const global void *vbq, 
        const global block_q8_1 *bq8_1, 
        const int kbx, 
        const int iqs) {

    const global block_q8_0 *bq8_0 = (const global block_q8_0 *)vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; i++) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl(v, u, bq8_0->d, bq8_1->ds.x);
}

#else
inline float vec_dot_q8_0_q8_1(
        const global void *vbq, 
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
        const int kbx, 
        const int iqs) {

    const global block_q8_0 *bq8_0 = (const global block_q8_0 *)vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

    unroll_for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; i++) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(q8_1_qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl(v, u, bq8_0->d, (*q8_1_ds).x);
}
#endif

)";

const char g_codeVecDot_Q2_K[] = R"(
inline float vec_dot_q2_K_q8_1(
        const global void *vbq, 
#if !USE_SOA_Y
        const global block_q8_1 *bq8_1, 
#else
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
#endif
        const int kbx, 
        const int iqs) {

    const global block_q2_K *bq2_K = (const global block_q2_K *)vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

    const global uchar *scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);

    int u[QR2_K];
    float d8[QR2_K];

    unroll_for (int i = 0; i < QR2_K; i++) {
#if !USE_SOA_Y
        u[i] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = convert_float(bq8_1[bq8_offset + i].ds.x);
#else
        u[i] = get_int_b4(&q8_1_qs[(bq8_offset + i) * QK8_1], iqs % QI8_1);
        d8[i] = convert_float(q8_1_ds[bq8_offset + i].x);
#endif
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

)";

const char g_codeVecDot_Q3_K[] = R"(
inline float vec_dot_q3_K_q8_1(
        const global void *vbq, 
#if !USE_SOA_Y
        const global block_q8_1 *bq8_1, 
#else
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
#endif
        const int kbx, 
        const int iqs) {

    const global block_q3_K *bq3_K = (const global block_q3_K *)vbq + kbx;

    const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

    const float d = bq3_K->d;

    const int vl = get_int_b2(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;

    int u[QR3_K];
    float d8[QR3_K];

    unroll_for (int i = 0; i < QR3_K; i++) {
#if !USE_SOA_Y
        u[i] = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = convert_float(bq8_1[bq8_offset + i].ds.x);
#else
        u[i] = get_int_b4(&q8_1_qs[(bq8_offset + i) * QK8_1], iqs % QI8_1);
        d8[i] = convert_float(q8_1_ds[bq8_offset + i].x);
#endif
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
} 

)";

const char g_codeVecDot_Q4_K[] = R"(
inline float vec_dot_q4_K_q8_1(
        const global void *vbq, 
#if !USE_SOA_Y
        const global block_q8_1 *bq8_1, 
#else
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
#endif
        const int kbx, 
        const int iqs) {

    const global block_q4_K *bq4_K = (const global block_q4_K *)vbq + kbx;

    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs / 4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const global int *q4 = (const global int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    const global ushort *scales = (const global ushort *)bq4_K->scales;
    ushort aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uchar *sc = (const uchar *)aux;
    const uchar *m = sc + 2;

    for (int i = 0; i < QR4_K; i++) {
#if !USE_SOA_Y
        const global block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
        d8[i] = convert_float(bq8i->ds.x);

        const global int *q8 = (const global int *)bq8i->qs + ((iqs / 2) % 4);
#else
        d8[i] = convert_float(q8_1_ds[bq8_offset + i].x);
        const global int *q8 = 
            (const global int *)(&q8_1_qs[(bq8_offset + i) * QK8_1]) + (iqs / 2) % 4;
#endif
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

)";

const char g_codeVecDot_Q5_K[] = R"(
inline float vec_dot_q5_K_q8_1(
        const global void *vbq, 
#if !USE_SOA_Y
        const global block_q8_1 *bq8_1, 
#else
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
#endif
        const int kbx, 
        const int iqs) {

    const global block_q5_K *bq5_K = (const global block_q5_K *)vbq + kbx;

    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const global int *ql = (const global int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const global int *qh = (const global int *)(bq5_K->qh + 4 * ((iqs / 2) % 4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const global ushort *scales = (const global ushort *)bq5_K->scales;
    ushort aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uchar *sc = (const uchar *)aux;
    const uchar *m = sc + 2;

    unroll_for (int i = 0; i < QR5_K; i++) {
#if !USE_SOA_Y
        const global block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
        d8[i] = convert_float(bq8i->ds.x);

        const global int *q8 = (const global int *)bq8i->qs + ((iqs / 2) % 4);
#else
        d8[i] = convert_float(q8_1_ds[bq8_offset + i].x);
        const global int *q8 = 
            (const global int *)(&q8_1_qs[(bq8_offset + i) * QK8_1]) + (iqs / 2) % 4;
#endif
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

)";

const char g_codeVecDot_Q6_K[] = R"(
inline float vec_dot_q6_K_q8_1(
        const global void *vbq, 
#if !USE_SOA_Y
        const global block_q8_1 *bq8_1, 
#else
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
#endif
        const int kbx, 
        const int iqs) {

    const global block_q6_K *bq6_K = (const global block_q6_K *)vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;

    const global char *scales = bq6_K->scales + scale_offset;

    int u[QR6_K];
    float d8[QR6_K];

    unroll_for (int i = 0; i < QR6_K; i++) {
#if !USE_SOA_Y
        u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        d8[i] = convert_float(bq8_1[bq8_offset + 2 * i].ds.x);
#else
        u[i] = get_int_b4(&q8_1_qs[(bq8_offset + 2 * i) * QK8_1], iqs % QI8_1);
        d8[i] = convert_float(q8_1_ds[bq8_offset + 2 * i].x);
#endif
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
} 

)";

const char g_codeVecDot_Mxfp4[] = R"(
#if !USE_SOA_Y
inline float vec_dot_mxfp4_q8_1(
        const global void *vbq, 
        const global block_q8_1 *bq8_1, 
        const int kbx, 
        const int iqs) {

    const global block_mxfp4 *bq4 = (const global block_mxfp4 *)vbq + kbx;

    const global int *q8 = (const global int *)bq8_1->qs + iqs;

    int sumi = 0;
    unroll_for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; l++) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        sumi = IMAD(v.x, q8[l + 0], sumi);
        sumi = IMAD(v.y, q8[l + 4], sumi);
    }

    const float d = e8m0_to_fp32(bq4->e) * 0.5f * convert_float(bq8_1->ds.x);
    return d * sumi;
} 

#else
inline float vec_dot_mxfp4_q8_1(
        const global void *vbq, 
        const global char *q8_1_qs,
        const global half2 *q8_1_ds,
        const int kbx, 
        const int iqs) {

    const global block_mxfp4 *bq4 = (const global block_mxfp4 *)vbq + kbx;

    const global int *q8 = (const global int *)q8_1_qs + iqs;

    int sumi = 0;
    unroll_for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; l++) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        sumi = IMAD(v.x, q8[l + 0], sumi);
        sumi = IMAD(v.y, q8[l + 4], sumi);
    }

    const float d = e8m0_to_fp32(bq4->e) * 0.5f * convert_float((*q8_1_ds).x);
    return d * sumi;
} 
#endif

)";

//
//    VecDotImpl
//

const char g_codeVecDotImpl_Q4_0[] = R"(
inline float vec_dot_q4_0_q8_1_impl(
        const int *v, 
        const int *u, 
        const float d4, 
        const half2 ds8) {

    int sumi = 0;

    unroll_for (int i = 0; i < VDR; i++) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = IMAD(vi0, u[2 * i + 0], sumi);
        sumi = IMAD(vi1, u[2 * i + 1], sumi);
    }

    const float2 ds8f = convert_float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8 * VDR / QI4_0) * ds8f.y);
} 

)";

const char g_codeVecDotImpl_Q4_1[] = R"(
inline float vec_dot_q4_1_q8_1_impl(
        const int *v, 
        const int *u, 
        const half2 dm4, 
        const half2 ds8) {

    int sumi = 0;

    unroll_for (int i = 0; i < VDR; i++) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = IMAD(vi0, u[2 * i + 0], sumi);
        sumi = IMAD(vi1, u[2 * i + 1], sumi);
    }

    const float2 tmp = convert_float2(dm4 * ds8);
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;

    // scale second part of sum to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (VDR * QR4_1));
}

)";

const char g_codeVecDotImpl_Q5_0[] = R"(
inline float vec_dot_q5_0_q8_1_impl(
        const int *vl, 
        const int *vh, 
        const int *u, 
        const float d5, 
        const half2 ds8) {

    int sumi = 0;

    unroll_for (int i = 0; i < VDR; i++) {
        int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;  // lower 4 qs bits, still need qh as 5th bits
        vi0 |= (vh[i] << 4) & 0x00000010;     // 0 ->  4
        vi0 |= (vh[i] << 11) & 0x00001000;    // 1 -> 12
        vi0 |= (vh[i] << 18) & 0x00100000;    // 2 -> 20
        vi0 |= (vh[i] << 25) & 0x10000000;    // 3 -> 28
        sumi = IMAD(vi0, u[2 * i + 0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;  // upper 4 qs bits, still need qh as 5th bits
        vi1 |= (vh[i] >> 12) & 0x00000010;    // 16 ->  4
        vi1 |= (vh[i] >> 5) & 0x00001000;     // 17 -> 12
        vi1 |= (vh[i] << 2) & 0x00100000;     // 18 -> 20
        vi1 |= (vh[i] << 9) & 0x10000000;     // 19 -> 28
        sumi = IMAD(vi1, u[2 * i + 1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = convert_float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16 * VDR / QI5_0) * ds8f.y);
}
 
)";

const char g_codeVecDotImpl_Q5_1[] = R"(
inline float vec_dot_q5_1_q8_1_impl(
        const int *vl, 
        const int *vh, 
        const int *u, 
        const half2 dm5, 
        const half2 ds8) {

    int sumi = 0;

    unroll_for (int i = 0; i < VDR; i++) {
        int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;  // lower 4 qs bits, still need qh as 5th bits
        vi0 |= (vh[i] << 4) & 0x00000010;     // 0 ->  4
        vi0 |= (vh[i] << 11) & 0x00001000;    // 1 -> 12
        vi0 |= (vh[i] << 18) & 0x00100000;    // 2 -> 20
        vi0 |= (vh[i] << 25) & 0x10000000;    // 3 -> 28
        sumi = IMAD(vi0, u[2 * i + 0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;  // upper 4 qs bits, still need qh as 5th bits
        vi1 |= (vh[i] >> 12) & 0x00000010;    // 16 ->  4
        vi1 |= (vh[i] >> 5) & 0x00001000;     // 17 -> 12
        vi1 |= (vh[i] << 2) & 0x00100000;     // 18 -> 20
        vi1 |= (vh[i] << 9) & 0x10000000;     // 19 -> 28
        sumi = IMAD(vi1, u[2 * i + 1], sumi); // SIMD dot product of quantized values
    }

    const float2 tmp = convert_float2(dm5 * ds8);
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;

    // scale second part of sum to compensate for multiple threads adding it
    return sumi * d5d8 + m5s8 / (QI5_1 / VDR);
} 

)";

const char g_codeVecDotImpl_Q8_0[] = R"(
inline float vec_dot_q8_0_q8_1_impl(
        const int *v, 
        const int *u, 
        const half d8_0, 
        const half d8_1) {

    int sumi = 0;

    unroll_for (int i = 0; i < VDR; i++) {
        // SIMD dot product of quantized values
        sumi = IMAD(v[i], u[i], sumi);
    }

    return convert_float(d8_0) * convert_float(d8_1)  * convert_float(sumi);
} 

)";

const char g_codeVecDotImpl_Q8_1[] = R"(
inline float vec_dot_q8_1_q8_1_impl(
        const int *v, 
        const int *u, 
        const half2 dm8, 
        const half2 ds8) {

    int sumi = 0;

    unroll_for (int i = 0; i < VDR; i++) {
        // SIMD dot product of quantized values
        sumi = IMAD(v[i], u[i], sumi);
    }

    const float2 tmp = convert_float2(dm8 * ds8);
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;

    // scale second part of sum by QI8_1 / VDR to compensate for multiple threads adding it
    return sumi * d8d8 + m8s8 / (QI8_1 / VDR);
}

)";

const char g_codeVecDotImpl_Q2_K[] = R"(
inline float vec_dot_q2_K_q8_1_impl_mmvq(
        const int v, 
        const int *u, 
        const global uchar *scales,
        const half2 dm2, 
        const float *d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    unroll_for (int i = 0; i < QR2_K; i++) {
        const int sc = scales[2 * i];

        const int vi = (v >> (2 * i)) & 0x03030303;

        sumf_d += d8[i] * (IMAD(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        // multiply constant q2_K part with sum of q8_1 values
        sumf_m += d8[i] * IMAD(m, u[i], 0); 
    }

    const float2 dm2f = convert_float2(dm2);

    return dm2f.x * sumf_d - dm2f.y * sumf_m;
} 

)";

const char g_codeVecDotImpl_Q3_K[] = R"(
inline float vec_dot_q3_K_q8_1_impl_mmvq(
        const int vl, 
        const int vh, 
        const int *u, 
        const global uchar *scales,
        const int scale_offset, 
        const float d3, 
        const float *d8) {

    float sumf = 0.0f;

    unroll_for (int i = 0; i < QR3_K; i++) {
        const int isc = scale_offset + 2 * i;

        const int isc_low = isc % (QK_K / 32);
        const int sc_shift_low = 4 * (isc / (QK_K / 32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K / 64);
        const int sc_shift_high = 2 * (isc / (QK_K / 64));
        const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = ISUB_SAT(vil, vih);

        sumf += d8[i] * (IMAD(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
} 

)";

const char g_codeVecDotImpl_Q4_K[] = R"(
inline float vec_dot_q4_K_q8_1_impl_vmmq(
        const int *v, 
        const int *u, 
        const uchar *sc,
        const uchar *m, 
        const half2 dm4, 
        const float *d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    unroll_for (int i = 0; i < QR4_K; i++) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;

        const int dot1 = IMAD(v1i, u[2 * i + 1], IMAD(v0i, u[2 * i + 0], 0)); // SIMD dot product
        const int dot2 = IMAD(0x01010101, u[2 * i + 1], IMAD(0x01010101, u[2 * i + 0], 0)); // sum of u

        // multiply constant part of q4_K with sum of q8_1 values
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  
    }

    const float2 dm4f = convert_float2(dm4);

    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

)";

const char g_codeVecDotImpl_Q5_K[] = R"(
inline float vec_dot_q5_K_q8_1_impl_vmmq(
        const int *vl, 
        const int *vh, 
        const int *u, 
        const uchar *sc,
        const uchar *m, 
        const half2 dm5, 
        const float *d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    unroll_for (int i = 0; i < QR5_K; i++) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = IMAD(v0i, u[2 * i + 0], IMAD(v1i, u[2 * i + 1], 0)); // SIMD dot product
        const int dot2 = IMAD(0x01010101, u[2 * i + 0], IMAD(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm5f = convert_float2(dm5);

    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

)";

const char g_codeVecDotImpl_Q6_K[] = R"(
inline float vec_dot_q6_K_q8_1_impl_mmvq(
        const int vl, 
        const int vh, 
        const int *u, 
        const global char *scales,
        const float d, 
        const float *d8) {

    float sumf = 0.0f;

    unroll_for (int i = 0; i < QR6_K; i++) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = ISUB_SAT((vil | vih), 0x20202020); // vi = (vil | vih) - 32
        sumf += d8[i] * (IMAD(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d * sumf;
}

)";

//
//    VecDotMmImpl
//

const char g_codeVecDotMmImpl_Q2_K[] = R"(
inline float vec_dot_q2_K_q8_1_impl_mmq(
        const int NS8,
        const local int *v, 
        const local int *u, 
        const local half2 *dm2, 
        const float d8, 
        const local half2 *s8) {

    float sumf = 0.0f;
    float sumf_d8 = 0.0f;

    unroll_for (int i0 = 0; i0 < QR2_K * VDR_Q2_K_Q8_1_MMQ; i0 += QI8_1) {
        const float2 dm2f0 = convert_float2(dm2[i0 / (QI8_1 / 2) + 0]);
        int sumi_d0 = 0;

        const float2 dm2f1 = convert_float2(dm2[i0 / (QI8_1 / 2) + 1]);
        int sumi_d1 = 0;

        unroll_for (int i = i0; i < i0 + QI8_1 / 2; i++) {
            sumi_d0 = IMAD(v[i], u[i], sumi_d0);
        }
        sumf_d8 += dm2f0.x * sumi_d0;

        unroll_for (int i = i0 + QI8_1 / 2; i < i0 + QI8_1; i++) {
            sumi_d1 = IMAD(v[i], u[i], sumi_d1);
        }
        sumf_d8 += dm2f1.x * sumi_d1;

        if (i0 / QI8_1 < NS8) {
            const float2 s8f = convert_float2(s8[i0 / QI8_1]);
            sumf -= dm2f0.y * s8f.x;
            sumf -= dm2f1.y * s8f.y;
        } else {
            int sumi_m0 = 0;
            unroll_for (int i = i0; i < i0 + QI8_1 / 2; i++) {
                sumi_m0 = IMAD(0x01010101, u[i], sumi_m0);
            }
            sumf_d8 -= dm2f0.y * sumi_m0;

            int sumi_m1 = 0;
            unroll_for (int i = i0 + QI8_1 / 2; i < i0 + QI8_1; i++) {
                sumi_m1 = IMAD(0x01010101, u[i], sumi_m1);
            }
            sumf_d8 -= dm2f1.y * sumi_m1;
        }
    }

    return sumf + d8 * sumf_d8;
}

)";

const char g_codeVecDotMmImpl_Q3_K[] = R"(
inline float vec_dot_q3_K_q8_1_impl_mmq(
        const local int *v, 
        const local int *u, 
        const char *scales,
        const float d3, 
        const float d8) {

    int sumi = 0;

    unroll_for (int i0 = 0; i0 < QR3_K * VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1 / 2) {
        int sumi_sc = 0;

        unroll_for (int i = i0; i < i0 + QI8_1 / 2; i++) {
            sumi_sc = IMAD(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1 / 2)];
    }

    return d3 * d8 * sumi;
} 

)";

const char g_codeVecDotMmImpl_Q4_K[] = R"(
inline float vec_dot_q4_K_q8_1_impl_mmq(
        const local int *v, 
        const local int *u, 
        const uchar *sc,
        const uchar *m, 
        const half2 dm4, 
        const local half2 *ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    unroll_for (int i = 0; i < QR4_K * VDR_Q4_K_Q8_1_MMQ / QI8_1; i++) {
        int sumi_d = 0;

        unroll_for (int j = 0; j < QI8_1; j++) {
            sumi_d = IMAD((v[j] >> (4 * i)) & 0x0F0F0F0F, u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = convert_float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y * m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = convert_float2(dm4);

    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

)";

const char g_codeVecDotMmImpl_Q5_K[] = R"(
inline float vec_dot_q5_K_q8_1_impl_mmq(
        const local int *v, 
        const local int *u, 
        const uchar *sc,
        const uchar *m, 
        const half2 dm4, 
        const local half2 *ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    unroll_for (int i = 0; i < QR5_K * VDR_Q5_K_Q8_1_MMQ / QI8_1; i++) {
        int sumi_d = 0;

        unroll_for (int j = 0; j < QI8_1; j++) {
            sumi_d = IMAD(v[i * QI8_1 + j], u[i * QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = convert_float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y * m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = convert_float2(dm4);

    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

)";

const char g_codeVecDotMmImpl_Q6_K[] = R"(
inline float vec_dot_q6_K_q8_1_impl_mmq(
        const int *v, 
        const int *u, 
        const char *sc,
        const float d6, 
        const local float *d8) {

    float sumf_d = 0.0f;

    const int sc_packed = get_int_b4(sc, 0);
    const char *sc_reg = (const char *)&sc_packed;

    unroll_for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

        unroll_for (int i = i0; i < i0 + 2; i++) {
            sumi_d.x = IMAD(v[2 * i + 0], u[2 * i + 0], sumi_d.x); // SIMD dot product
            sumi_d.x = IMAD(v[2 * i + 1], u[2 * i + 1], sumi_d.x); // SIMD dot product

            sumi_d.y = IMAD(v[2 * i + 4], u[2 * i + 4], sumi_d.y); // SIMD dot product
            sumi_d.y = IMAD(v[2 * i + 5], u[2 * i + 5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0 / 4] * (sc_reg[i0 / 2 + 0] * sumi_d.x + sc_reg[i0 / 2 + 1] * sumi_d.y);
    }

    return d6 * sumf_d;
}

)";

} // namespace

const char *VecDotQuantCommonCode() {
    return g_codeVecDotQuantCommon;
}

//
//    VecDotDefs
//

const char *VecDotDefs_Q4_0_Code() {
    return g_codeVecDotDefs_Q4_0;
}

const char *VecDotDefs_Q4_1_Code() {
    return g_codeVecDotDefs_Q4_1;
}

const char *VecDotDefs_Q5_0_Code() {
    return g_codeVecDotDefs_Q5_0;
}

const char *VecDotDefs_Q5_1_Code() {
    return g_codeVecDotDefs_Q5_1;
}

const char *VecDotDefs_Q8_0_Code() {
    return g_codeVecDotDefs_Q8_0;
}

const char *VecDotDefs_Q2_K_Code() {
    return g_codeVecDotDefs_Q2_K;
}

const char *VecDotDefs_Q3_K_Code() {
    return g_codeVecDotDefs_Q3_K;
}

const char *VecDotDefs_Q4_K_Code() {
    return g_codeVecDotDefs_Q4_K;
}

const char *VecDotDefs_Q5_K_Code() {
    return g_codeVecDotDefs_Q5_K;
}

const char *VecDotDefs_Q6_K_Code() {
    return g_codeVecDotDefs_Q6_K;
}

const char *VecDotDefs_Mxfp4_Code() {
    return g_codeVecDotDefs_Mxfp4;
}

//
//
//    VecDot

const char *VecDot_Q4_0_Code() {
    return g_codeVecDot_Q4_0;
}

const char *VecDot_Q4_1_Code() {
    return g_codeVecDot_Q4_1;
}

const char *VecDot_Q5_0_Code() {
    return g_codeVecDot_Q5_0;
}

const char *VecDot_Q5_1_Code() {
    return g_codeVecDot_Q5_1;
}

const char *VecDot_Q8_0_Code() {
    return g_codeVecDot_Q8_0;
}

const char *VecDot_Q2_K_Code() {
    return g_codeVecDot_Q2_K;
}

const char *VecDot_Q3_K_Code() {
    return g_codeVecDot_Q3_K;
}

const char *VecDot_Q4_K_Code() {
    return g_codeVecDot_Q4_K;
}

const char *VecDot_Q5_K_Code() {
    return g_codeVecDot_Q5_K;
}

const char *VecDot_Q6_K_Code() {
    return g_codeVecDot_Q6_K;
}

const char *VecDot_Mxfp4_Code() {
    return g_codeVecDot_Mxfp4;
}

//
//    VecDotImpl
//

const char *VecDotImpl_Q4_0_Code() {
    return g_codeVecDotImpl_Q4_0;
}

const char *VecDotImpl_Q4_1_Code() {
    return g_codeVecDotImpl_Q4_1;
}

const char *VecDotImpl_Q5_0_Code() {
    return g_codeVecDotImpl_Q5_0;
}

const char *VecDotImpl_Q5_1_Code() {
    return g_codeVecDotImpl_Q5_1;
}

const char *VecDotImpl_Q8_0_Code() {
    return g_codeVecDotImpl_Q8_0;
}

const char *VecDotImpl_Q8_1_Code() {
    return g_codeVecDotImpl_Q8_1;
}

const char *VecDotImpl_Q2_K_Code() {
    return g_codeVecDotImpl_Q2_K;
}

const char *VecDotImpl_Q3_K_Code() {
    return g_codeVecDotImpl_Q3_K;
}

const char *VecDotImpl_Q4_K_Code() {
    return g_codeVecDotImpl_Q4_K;
}

const char *VecDotImpl_Q5_K_Code() {
    return g_codeVecDotImpl_Q5_K;
}

const char *VecDotImpl_Q6_K_Code() {
    return g_codeVecDotImpl_Q6_K;
}

//
//    VecDotMmImpl
//

const char *VecDotMmImpl_Q2_K_Code() {
    return g_codeVecDotMmImpl_Q2_K;
}

const char *VecDotMmImpl_Q3_K_Code() {
    return g_codeVecDotMmImpl_Q3_K;
}

const char *VecDotMmImpl_Q4_K_Code() {
    return g_codeVecDotMmImpl_Q4_K;
}

const char *VecDotMmImpl_Q5_K_Code() {
    return g_codeVecDotMmImpl_Q5_K;
}

const char *VecDotMmImpl_Q6_K_Code() {
    return g_codeVecDotMmImpl_Q6_K;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

