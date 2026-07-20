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
//    MulMatVecV2Common
//

const char g_codeMmvCommon[] = R"(
inline FLOAT_TYPE dot4(
        const FLOAT_TYPE x0,
        const FLOAT_TYPE y0,
        const FLOAT_TYPE x1,
        const FLOAT_TYPE y1,
        const FLOAT_TYPE x2,
        const FLOAT_TYPE y2,
        const FLOAT_TYPE x3,
        const FLOAT_TYPE y3) {
    FLOAT_TYPE sum = x0 * y0;
    sum = fma(x1, y1, sum);
    sum = fma(x2, y2, sum);
    sum = fma(x3, y3, sum);
    return sum;
}

inline FLOAT_TYPE dot8_acc(
        const FLOAT_TYPE x0,
        const FLOAT_TYPE y0,
        const FLOAT_TYPE x1,
        const FLOAT_TYPE y1,
        const FLOAT_TYPE x2,
        const FLOAT_TYPE y2,
        const FLOAT_TYPE x3,
        const FLOAT_TYPE y3,
        const FLOAT_TYPE x4,
        const FLOAT_TYPE y4,
        const FLOAT_TYPE x5,
        const FLOAT_TYPE y5,
        const FLOAT_TYPE x6,
        const FLOAT_TYPE y6,
        const FLOAT_TYPE x7,
        const FLOAT_TYPE y7,
        FLOAT_TYPE sum) {
    sum = fma(x0, y0, sum);
    sum = fma(x1, y1, sum);
    sum = fma(x2, y2, sum);
    sum = fma(x3, y3, sum);
    sum = fma(x4, y4, sum);
    sum = fma(x5, y5, sum);
    sum = fma(x6, y6, sum);
    sum = fma(x7, y7, sum);
    return sum;
}

inline FLOAT_TYPE dot16(
        const FLOAT_TYPE x0,
        const FLOAT_TYPE y0,
        const FLOAT_TYPE x1,
        const FLOAT_TYPE y1,
        const FLOAT_TYPE x2,
        const FLOAT_TYPE y2,
        const FLOAT_TYPE x3,
        const FLOAT_TYPE y3,
        const FLOAT_TYPE x4,
        const FLOAT_TYPE y4,
        const FLOAT_TYPE x5,
        const FLOAT_TYPE y5,
        const FLOAT_TYPE x6,
        const FLOAT_TYPE y6,
        const FLOAT_TYPE x7,
        const FLOAT_TYPE y7,
        const FLOAT_TYPE x8,
        const FLOAT_TYPE y8,
        const FLOAT_TYPE x9,
        const FLOAT_TYPE y9,
        const FLOAT_TYPE xa,
        const FLOAT_TYPE ya,
        const FLOAT_TYPE xb,
        const FLOAT_TYPE yb,
        const FLOAT_TYPE xc,
        const FLOAT_TYPE yc,
        const FLOAT_TYPE xd,
        const FLOAT_TYPE yd,
        const FLOAT_TYPE xe,
        const FLOAT_TYPE ye,
        const FLOAT_TYPE xf,
        const FLOAT_TYPE yf) {
    FLOAT_TYPE sum = x0 * y0;
    sum = fma(x1, y1, sum);
    sum = fma(x2, y2, sum);
    sum = fma(x3, y3, sum);
    sum = fma(x4, y4, sum);
    sum = fma(x5, y5, sum);
    sum = fma(x6, y6, sum);
    sum = fma(x7, y7, sum);
    sum = fma(x8, y8, sum);
    sum = fma(x9, y9, sum);
    sum = fma(xa, ya, sum);
    sum = fma(xb, yb, sum);
    sum = fma(xc, yc, sum);
    sum = fma(xd, yd, sum);
    sum = fma(xe, ye, sum);
    sum = fma(xf, yf, sum);
    return sum;
}

#define DOT4 dot4
#define DOT8_ACC dot8_acc
#define DOT16 dot16

#define CAST_CONST_PACKED16(type, data) \
    const global type##_packed16 *data##_packed16 = (const global type##_packed16 *)(data)
#define CAST_CONST_PACKED32(type, data) \
    const global type##_packed32 *data##_packed32 = (const global type##_packed32 *)(data)

)";

//
//    MulMatVecV2Impl
//

const char g_codeMmvImpl_Q4_0[] = R"(
inline float2 dequantize(
        const global block_q4_0 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    const uint vui = (uint)data_a[a_offset + ib].qs[iqs];
    return (float2)(vui & 0xF, vui >> 4) - 8.0f;
}

inline float4 dequantize4(
        const global block_q4_0 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    CAST_CONST_PACKED16(block_q4_0, data_a);
    const uint vui = (uint)data_a_packed16[a_offset + ib].qs[iqs/2];
    return (float4)(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f;
} 

inline float2 get_dm(
        const global block_q4_0 *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)((float)data_a[a_offset + ib].d, 0.0f);
} 

)";

const char g_codeMmvImpl_Q4_1[] = R"(
inline float2 dequantize(
        const global block_q4_1 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    const uint vui = (uint)data_a[a_offset + ib].qs[iqs];
    return (float2)(vui & 0xF, vui >> 4);
}

inline float4 dequantize4(
        const global block_q4_1 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    CAST_CONST_PACKED16(block_q4_1, data_a);
    const uint vui = (uint)data_a_packed16[a_offset + ib].qs[iqs/2];
    return (float4)(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12);
} 

inline float2 get_dm(
        const global block_q4_1 *data_a,
        const uint ib, 
        const uint a_offset) {
    CAST_CONST_PACKED32(block_q4_1, data_a);
    return convert_float2(data_a_packed32[a_offset + ib].dm);
} 

)";

const char g_codeMmvImpl_Q5_0[] = R"(
inline float2 dequantize(
        const global block_q5_0 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    const uint uint_qh = ((uint)data_a[a_offset + ib].qh[1] << 16) | data_a[a_offset + ib].qh[0];
    const int2 qh = (int2)(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = (uint)data_a[a_offset + ib].qs[iqs];
    return (float2)((vui & 0xF) | qh.x, (vui >> 4) | qh.y) - 16.0f;
}

inline float4 dequantize4(
        const global block_q5_0 *data_a,
        uint ib, 
        uint iqs, 
        uint a_offset) {
    CAST_CONST_PACKED16(block_q5_0, data_a);
    const uint uint_qh = 
        ((uint)data_a_packed16[a_offset + ib].qh[1] << 16) | 
            data_a_packed16[a_offset + ib].qh[0];
    const int2 qh0 = (int2)(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const int2 qh1 = (int2)(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = (uint)data_a_packed16[a_offset + ib].qs[iqs/2];
    return (float4)(
        (vui & 0xF) | qh0.x, 
        ((vui >> 4) & 0xF) | qh0.y, 
        ((vui >> 8) & 0xF) | qh1.x, 
        (vui >> 12) | qh1.y) - 16.0f;
} 

inline float2 get_dm(
        const global block_q5_0 *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)((float)data_a[a_offset + ib].d, 0.0f);
} 

)";

const char g_codeMmvImpl_Q5_1[] = R"(
inline float2 dequantize(
        const global block_q5_1 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    const uint uint_qh = data_a[a_offset + ib].qh;
    const int2 qh = (int2)(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const uint vui = (uint)data_a[a_offset + ib].qs[iqs];
    return (float2)((vui & 0xF) | qh.x, (vui >> 4) | qh.y);
}

inline float4 dequantize4(
        const global block_q5_1 *data_a,
        uint ib, 
        uint iqs, 
        uint a_offset) {
    CAST_CONST_PACKED16(block_q5_1, data_a);
    const uint uint_qh = data_a_packed16[a_offset + ib].qh;
    const int2 qh0 = (int2)(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
    const int2 qh1 = (int2)(((uint_qh >> (iqs + 1)) << 4) & 0x10, (uint_qh >> (iqs + 13)) & 0x10);
    const uint vui = (uint)data_a_packed16[a_offset + ib].qs[iqs/2];
    return (float4)(
        (vui & 0xF) | qh0.x, 
        ((vui >> 4) & 0xF) | qh0.y, 
        ((vui >> 8) & 0xF) | qh1.x, 
        (vui >> 12) | qh1.y);
} 

inline float2 get_dm(
        const global block_q5_1 *data_a,
        const uint ib, 
        const uint a_offset) {
    CAST_CONST_PACKED32(block_q5_1, data_a);
    return convert_float2(data_a_packed32[a_offset + ib].dm);
} 

)";

const char g_codeMmvImpl_Q8_0[] = R"(
inline float2 dequantize(
        const global block_q8_0 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    return (float2)((int)data_a[a_offset + ib].qs[iqs], (int)data_a[a_offset + ib].qs[iqs + 1]);
}

inline float4 dequantize4(
        const global block_q8_0 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    CAST_CONST_PACKED16(block_q8_0, data_a);
    const char2 v0 = as_char4((int)data_a_packed16[a_offset + ib].qs[iqs/2]).xy;
    const char2 v1 = as_char4((int)data_a_packed16[a_offset + ib].qs[iqs/2 + 1]).xy;
    return (float4)(v0.x, v0.y, v1.x, v1.y);
} 

inline float2 get_dm(
        const global block_q8_0 *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)((float)data_a[a_offset + ib].d, 0.0f);
} 

)";

const char g_codeMmvImpl_Q2_K[] = R"(
inline float2 dequantize(
        const global block_q2_K *data_a,
        const uint ib, 
        uint iqs, 
        const uint a_offset) {
    iqs /= 2;
    const uint qsi = (iqs / 64) * 32 + (iqs % 16) * 2; // 0,2,4..30
    const uint scalesi = iqs / 8;                      // 0..15
    const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

    const uint2 qs = (uint2)(data_a[a_offset + ib].qs[qsi], data_a[a_offset + ib].qs[qsi + 1]);
    const uint scales = data_a[a_offset + ib].scales[scalesi];
    const float2 dm = convert_float2(data_a[a_offset + ib].dm);

    return dm.x * (float)(scales & 0xF) * convert_float2((qs >> qsshift) & 3) - dm.y * (float)(scales >> 4);
}

inline float2 get_dm(
        const global block_q2_K *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)(1.0f, 0.0f);
} 

)";

const char g_codeMmvImpl_Q3_K[] = R"(
inline float2 dequantize(
        const global block_q3_K *data_a,
        const uint ib, 
        uint iqs, 
        const uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                     // 0,1
    const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
    const uint hmi = (iqs % 16) * 2;             // 0,2,4..30
    const uint j = (iqs % 64) / 4;               // 0..3
    const uint is = iqs / 8;                     // 0..15
    const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
    const uint qsshift = halfsplit * 2;          // 0,2,4,6
    const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

    const char us = 
        (char)(((data_a[a_offset + ib].scales[is % 8] >> (4 * (int)(is / 8))) & 0xF) | 
            (((data_a[a_offset + ib].scales[8 + (is % 4)] >> (2 * (int)(is / 4))) & 3) << 4));
    const float dl = (float)data_a[a_offset + ib].d * (float)(us - 32);

    return (float2)(
        dl * (float)((char)((data_a[a_offset + ib].qs[qsi] >> qsshift) & 3) - 
            (((data_a[a_offset + ib].hmask[hmi] & m) != 0) ? 0 : 4)),
        dl * (float)((char)((data_a[a_offset + ib].qs[qsi + 1] >> qsshift) & 3) - 
            (((data_a[a_offset + ib].hmask[hmi + 1] & m) != 0) ? 0 : 4)));
}

inline float2 get_dm(
        const global block_q3_K *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)(1.0f, 0.0f);
} 

)";

const char g_codeMmvImpl_Q4_K[] = R"(
inline float2 dequantize(
        const global block_q4_K *data_a,
        const uint ib, 
        uint iqs, 
        const uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

    const float2 loadd = convert_float2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uchar sc = 
        (uchar)((data_a[a_offset + ib].scales[scidx0] & 0xF) | 
            ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uchar mbyte = 
        (uchar)((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0 | 
            ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return (float2)(
        fma(d, (float)((data_a[a_offset + ib].qs[qsi] >> (b * 4)) & 0xF), m),
        fma(d, (float)((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF), m));
}

inline float2 get_dm(
        const global block_q4_K *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)(1.0f, 0.0f);
} 

)";

const char g_codeMmvImpl_Q5_K[] = R"(
inline float2 dequantize(
        const global block_q5_K *data_a,
        const uint ib, 
        uint iqs, 
        const uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 32;                   // 0,1,2,3
    const uint b = (iqs % 32) / 16;            // 0,1
    const uint is = 2 * n + b;                 // 0..7
    const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
    const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

    const uchar hm = (uchar)(1 << (iqs / 16));

    const float2 loadd = convert_float2(data_a[a_offset + ib].dm);

    const uint scidx0 = (is < 4) ? is : (is + 4);
    const uint scidx1 = (is < 4) ? is : (is - 4);
    const uint scidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint scidxshift1 = (is < 4) ? 0 : 2;
    const uint mbidx0 = is + 4;
    const uint mbidx1 = (is < 4) ? is + 4 : is;
    const uint mbidxmask0 = (is < 4) ? 0xF : 0xF0;
    const uint mbidxshift0 = (is < 4) ? 0 : 4;
    const uint mbidxmask1 = (is < 4) ? 0x30 : 0xC0;
    const uint mbidxshift1 = (is < 4) ? 0 : 2;

    const uchar sc = 
        (uchar)((data_a[a_offset + ib].scales[scidx0] & 0xF) | 
            ((data_a[a_offset + ib].scales[scidx1] & scidxmask1) >> scidxshift1));
    const uchar mbyte = 
        (uchar)(((data_a[a_offset + ib].scales[mbidx0] & mbidxmask0) >> mbidxshift0) | 
            ((data_a[a_offset + ib].scales[mbidx1] & mbidxmask1) >> mbidxshift1));

    const float d = loadd.x * sc;
    const float m = -loadd.y * mbyte;

    return (float2)(
        fma(
            d, 
            (float)((data_a[a_offset + ib].qs[qsi] >> (b * 4)) & 0xF) + 
                (float)(((data_a[a_offset + ib].qh[qhi] & hm) != 0) ? 16 : 0), 
            m),
        fma(
            d, 
            (float)((data_a[a_offset + ib].qs[qsi + 1] >> (b * 4)) & 0xF) + 
                (float)(((data_a[a_offset + ib].qh[qhi + 1] & hm) != 0) ? 16 : 0), 
            m));
}

inline float2 get_dm(
        const global block_q5_K *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)(1.0f, 0.0f);
} 

)";

const char g_codeMmvImpl_Q6_K[] = R"(
inline float2 dequantize(
        const global block_q6_K *data_a,
        const uint ib, 
        uint iqs, 
        const uint a_offset) {
    iqs /= 2;
    const uint n = iqs / 64;                    // 0,1
    const uint b = (iqs % 64) / 32;             // 0,1
    const uint is_b = (iqs % 16) / 8;           // 0,1
    const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
    const uint is = 8 * n + qhshift + is_b;     // 0..15
    const uint qsi = n * 64 + (iqs % 32) * 2;   // 0,2,4..126
    const uint qhi = n * 32 + (iqs % 16) * 2;   // 0,2,4..62

    const float dscale = (float)data_a[a_offset + ib].d * (float)data_a[a_offset + ib].scales[is];

    return (float2)(
        dscale * (float)((char)(((data_a[a_offset + ib].ql[qsi] >> (b * 4)) & 0xF) | 
            (((data_a[a_offset + ib].qh[qhi] >> qhshift) & 3) << 4)) - 32),
        dscale * (float)((char)(((data_a[a_offset + ib].ql[qsi + 1] >> (b * 4)) & 0xF) | 
            (((data_a[a_offset + ib].qh[qhi + 1] >> qhshift) & 3) << 4)) - 32));
}

inline float2 get_dm(
        const global block_q6_K *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)(1.0f, 0.0f);
} 

)";

const char g_codeMmvImpl_Mxfp4[] = R"(
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

inline float2 dequantize(
        const global block_mxfp4 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    const uint vui = (uint)data_a[a_offset + ib].qs[iqs];
    return (float2)(kvalues_mxfp4[vui & 0xF], kvalues_mxfp4[vui >> 4]) * 0.5f;
}

inline float4 dequantize4(
        const global block_mxfp4 *data_a,
        const uint ib, 
        const uint iqs, 
        const uint a_offset) {
    float2 v0 = dequantize(data_a, ib, iqs, a_offset);
    float2 v1 = dequantize(data_a, ib, iqs + 1, a_offset);
    return (float4)(v0.x, v0.y, v1.x, v1.y);
} 

inline float2 get_dm(
        const global block_mxfp4 *data_a,
        const uint ib, 
        const uint a_offset) {
    return (float2)(e8m0_to_fp32(data_a[a_offset + ib].e), 0.0f);
} 

)";

//
//    MulMatVec2
//

const char g_kernelCodeMmv[] = R"(
inline void iter(
        const global A_TYPE *data_a,
        const global B_TYPE *data_b,
        const uint ncols,
        const uint batch_stride_b,
        const uint a_offset,
        const uint b_offset,
        const uint y_offset,
        FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], 
        const uint first_row, 
        const uint num_rows, 
        const uint tid, 
        const uint i, 
        bool lastiter) {

    const global B_TYPE_V4 *data_b_v4 = (const global B_TYPE_V4 *)data_b;

    unroll_for (uint j = 0; j < NUM_COLS; j++) {
        const uint col = i * BLOCK_SIZE + K_PER_ITER * tid;
        const uint iqs = (col % QUANT_K) / QUANT_R; // quant index
        const uint iybs = col - col % QUANT_K;      // y block start index

#if K_PER_ITER == 8
#if QUANT_R == 2
        const float4 bv02 = 
            convert_float4(data_b_v4[(j * batch_stride_b + b_offset + iybs + iqs) / 4]);
        const float4 bv13 = 
            convert_float4(data_b_v4[(j * batch_stride_b + b_offset + iybs + iqs + y_offset) / 4]);
        const float4 bv0 = (float4)(bv02.x, bv13.x, bv02.y, bv13.y);
        const float4 bv1 = (float4)(bv02.z, bv13.z, bv02.w, bv13.w);
#else
        const float4 bv0 = 
            convert_float4(data_b_v4[(j * batch_stride_b + b_offset + iybs + iqs) / 4]);
        const float4 bv1 = 
            convert_float4(data_b_v4[(j * batch_stride_b + b_offset + iybs + iqs) / 4 + 1]);
#endif
#else
        // Check if the second of the pair of elements is OOB, and don't fetch B or accumulate it. 
        // We still fetch a pair of elements for A, which is fine for quantized formats since
        // they'll be within the same block. We should probably skip fetching the second element 
        // for F16/F32, but as of now we still do.
        const bool OOB = lastiter && (iybs + iqs + y_offset >= ncols);

        FLOAT_TYPE b0 = 0;
        FLOAT_TYPE b1 = 0;
        b0 = (FLOAT_TYPE)data_b[j * batch_stride_b + b_offset + iybs + iqs];
        if (!OOB) {
            b1 = (FLOAT_TYPE)data_b[j * batch_stride_b + b_offset + iybs + iqs + y_offset];
        }
#endif

        uint ibi = first_row * ncols;
        unroll_for (uint n = 0; n < num_rows; n++) {
            const uint ib = (ibi + col) / QUANT_K; // block index
            ibi += ncols;

#if K_PER_ITER == 8
            float4 v = dequantize4(data_a, ib, iqs, a_offset);
            float4 v2 = dequantize4(data_a, ib, iqs + (4 / QUANT_R), a_offset);

            const float2 dm = get_dm(data_a, ib, a_offset);
            if (dm.y != 0) { 
                // quant has min component
                v = v * dm.x + dm.y;
                v2 = v2 * dm.x + dm.y;
            }

            // matrix multiplication
            FLOAT_TYPE rowtmp = dot(bv0, v);
            rowtmp += dot(bv1, v2);

            if (dm.y == 0) {
                rowtmp *= dm.x;
            }

            temp[j][n] += rowtmp;
#else
            const float2 v = dequantize(data_a, ib, iqs, a_offset);

            // matrix multiplication
            temp[j][n] = fma((FLOAT_TYPE)v.x, b0, temp[j][n]);
            if (!OOB) {
                temp[j][n] = fma((FLOAT_TYPE)v.y, b1, temp[j][n]);
            }
#endif
        }
    }
}

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
        local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
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

    uint y_offset = (QUANT_R == 1) ? 1 : QUANT_K / 2;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS] = {{(FLOAT_TYPE)0}}; 

    uint num_iters = ncols / (K_PER_ITER * BLOCK_SIZE);
    if (num_iters * K_PER_ITER * BLOCK_SIZE + K_PER_ITER * tid < ncols) {
        num_iters++;
    }

#define ITER(idx, last) \
    iter(data_a, data_b, ncols, batch_stride_b, a_offset, b_offset, y_offset, \
        temp, first_row, num_rows, tid, (idx) * K_PER_ITER, last)

    int unroll_count = 4;
    uint unrolled_iters = num_iters & ~(unroll_count - 1);

#if K_PER_ITER == 2
    // If the K dimension is odd, we need lastiter==true on the last iteration
    // so OOB is computed correctly. Skip some unrolling to make that happen.
    if ((ncols & 1) != 0 &&
            unrolled_iters == num_iters &&
            unrolled_iters > 0) {
        unrolled_iters -= unroll_count;
    }
#endif

    uint i = 0;
    while (i < unrolled_iters) {
        // Manually partially unroll the loop
        unroll_for (uint k = 0; k < unroll_count; k++) {
            ITER(i, false);
            i++;
        }
    }

    unroll_count = 2;
    unrolled_iters = num_iters & ~(unroll_count - 1);

#if K_PER_ITER == 2
    if ((ncols & 1) != 0 &&
            unrolled_iters == num_iters &&
            unrolled_iters > 0) {
        unrolled_iters -= unroll_count;
    }
#endif

    while (i < unrolled_iters) {
        // Manually partially unroll the loop
        unroll_for (uint k = 0; k < unroll_count; k++) {
            ITER(i, false);
            i++;
        }
    }

    while (i < num_iters) {
        ITER(i, true);
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
kernel void mul_mat_vec(
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
    data_d += D_BASE / sizeof(float);

#ifndef USE_SUBGROUP_ADD_NO_SHMEM
    local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

    const uint first_row = NUM_ROWS * (GID_0 + GDIM_0 * GID_2);

    // do NUM_ROWS at a time, unless there aren't enough remaining rows
    if (first_row + NUM_ROWS <= stride_d) {
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
            NUM_ROWS);
    } else {
        if (first_row >= stride_d) {
            return;
        }
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
            stride_d - first_row);
    }
} 

)";

const char g_kernelCodeMmv_Q2_K[] = R"(
inline void calc_superblock(
        const global block_q2_K *data_a,
        const global B_TYPE *data_b,
        const uint batch_stride_b,
        local FLOAT_TYPE sccache1[2][BLOCK_SIZE / 16][16],
        local FLOAT_TYPE sccache2[2][BLOCK_SIZE / 16][16],
        FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], 
        uint *cselp,
        const uint a_offset, 
        const uint b_offset, 
        const uint itid, 
        const uint v_im, 
        const uint ix, 
        const uint q_offset, 
        const uint y_offset, 
        const uint i, 
        const uint num_blocks_per_row, 
        const uint first_row, 
        const uint num_rows, 
        const bool all_threads) {

    CAST_CONST_PACKED16(block_q2_K, data_a);
    const global B_TYPE_V2 *data_b_v2 = (const global B_TYPE_V2 *)data_b;

    const uint y_idx = i * QUANT_K + y_offset;

    uint csel = *cselp;

    unroll_for (uint n = 0; n < num_rows; n++) {
        const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
        csel ^= 1;

        if (!all_threads) { 
            // when we don't have enough blocks to use all threads
            if (i < num_blocks_per_row) {
                const uint scale = (uint)data_a[ib0 + i].scales[itid];
                sccache1[csel][ix][itid] = (FLOAT_TYPE)(scale & 0xF);
                sccache2[csel][ix][itid] = (FLOAT_TYPE)((scale >> 4) & 0xF);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (i >= num_blocks_per_row) {
                continue;
            }
        } else {
            const uint scale = (uint)data_a[ib0 + i].scales[itid];
            sccache1[csel][ix][itid] = (FLOAT_TYPE)(scale & 0xF);
            sccache2[csel][ix][itid] = (FLOAT_TYPE)((scale >> 4) & 0xF);
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const uint qs_u32 = 
            (uint)data_a_packed16[ib0 + i].qs[q_offset / 2] | 
                ((uint)data_a_packed16[ib0 + i].qs[q_offset / 2 + 8] << 16);
        const float4 qs_u32_0 = convert_float4(as_uchar4(qs_u32 & 0x03030303));
        const float4 qs_u32_2 = convert_float4(as_uchar4((qs_u32 >> 2) & 0x03030303));
        const float4 qs_u32_4 = convert_float4(as_uchar4((qs_u32 >> 4) & 0x03030303));
        const float4 qs_u32_6 = convert_float4(as_uchar4((qs_u32 >> 6) & 0x03030303));

        const FLOAT_TYPE_V2 dm = CONVERT_FLOAT_TYPE_V2(data_a[ib0 + i].dm);

        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            float2 b0 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 0]);
            float2 b16 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 8]);
            float2 b32 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 16]);
            float2 b48 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 24]);
            float2 b64 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 32]);
            float2 b80 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 40]);
            float2 b96 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 48]);
            float2 b112 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 56]);

            FLOAT_TYPE sum1 = (FLOAT_TYPE)0;
            FLOAT_TYPE sum2 = (FLOAT_TYPE)0;
            unroll_for (int l = 0; l < 2; l++) {
                sum1 = 
                    DOT8_ACC(
                        (FLOAT_TYPE)b0[l], sccache1[csel][ix][8 * v_im] * qs_u32_0[l],
                        (FLOAT_TYPE)b16[l], sccache1[csel][ix][1 + 8 * v_im] * qs_u32_0[l + 2],
                        (FLOAT_TYPE)b32[l], sccache1[csel][ix][2 + 8 * v_im] * qs_u32_2[l],
                        (FLOAT_TYPE)b48[l], sccache1[csel][ix][3 + 8 * v_im] * qs_u32_2[l + 2],
                        (FLOAT_TYPE)b64[l], sccache1[csel][ix][4 + 8 * v_im] * qs_u32_4[l],
                        (FLOAT_TYPE)b80[l], sccache1[csel][ix][5 + 8 * v_im] * qs_u32_4[l + 2],
                        (FLOAT_TYPE)b96[l], sccache1[csel][ix][6 + 8 * v_im] * qs_u32_6[l],
                        (FLOAT_TYPE)b112[l], sccache1[csel][ix][7 + 8 * v_im] * qs_u32_6[l + 2], 
                        sum1);
                sum2 = 
                    DOT8_ACC(
                        (FLOAT_TYPE)b0[l], sccache2[csel][ix][8 * v_im],
                        (FLOAT_TYPE)b16[l], sccache2[csel][ix][1 + 8 * v_im],
                        (FLOAT_TYPE)b32[l], sccache2[csel][ix][2 + 8 * v_im],
                        (FLOAT_TYPE)b48[l], sccache2[csel][ix][3 + 8 * v_im],
                        (FLOAT_TYPE)b64[l], sccache2[csel][ix][4 + 8 * v_im],
                        (FLOAT_TYPE)b80[l], sccache2[csel][ix][5 + 8 * v_im],
                        (FLOAT_TYPE)b96[l], sccache2[csel][ix][6 + 8 * v_im],
                        (FLOAT_TYPE)b112[l], sccache2[csel][ix][7 + 8 * v_im], 
                        sum2);
            }
            temp[j][n] = fma(dm.x, sum1, fma(-dm.y, sum2, temp[j][n]));
        }
    }

    *cselp = csel;
}

void compute_outputs(
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
        local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
#endif
        local FLOAT_TYPE sccache1[2][BLOCK_SIZE / 16][16],
        local FLOAT_TYPE sccache2[2][BLOCK_SIZE / 16][16],
        const uint first_row, 
        const uint num_rows) {

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

    const uint num_blocks_per_row = ncols / QUANT_K;

    // 16 threads are used to process each block
    const uint it_size = LDIM_0 / 16;
    const uint tid = LID_0;
    const uint itid = tid % 16;            // 0...15
    const uint ix = tid / 16;

    const uint v_im = itid / 8;            // 0 or 1. 0 computes 0..., 1 computes 128...
    const uint v_in = itid - 8 * v_im;     // 0...7

    const uint l0 = 2 * v_in;              // 0...15
    const uint q_offset = 32 * v_im + l0;
    const uint y_offset = 128 * v_im + l0;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS] = {{(FLOAT_TYPE)0}}; 

    uint csel = 0;

    const uint nbr_par_th = num_blocks_per_row % it_size;
    const uint nbr_all_th = num_blocks_per_row - nbr_par_th;
    uint i0 = 0;
    unroll_for ( ; i0 < nbr_all_th; i0 += it_size) {
        calc_superblock(
            data_a,
            data_b,
            batch_stride_b,
            sccache1,
            sccache2,
            temp, 
            &csel,
            a_offset, 
            b_offset, 
            itid, 
            v_im,  
            ix, 
            q_offset, 
            y_offset, 
            i0 + ix, 
            num_blocks_per_row, 
            first_row, 
            num_rows,  
            true);
    }
    calc_superblock(
        data_a,
        data_b,
        batch_stride_b,
        sccache1,
        sccache2,
        temp, 
        &csel,
        a_offset, 
        b_offset, 
        itid, 
        v_im, 
        ix, 
        q_offset, 
        y_offset, 
        i0 + ix, 
        num_blocks_per_row, 
        first_row, 
        num_rows, 
        false);

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
kernel void mul_mat_vec(
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
    data_d += D_BASE / sizeof(float);

#ifndef USE_SUBGROUP_ADD_NO_SHMEM
    local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

    local FLOAT_TYPE sccache1[2][BLOCK_SIZE / 16][16];
    local FLOAT_TYPE sccache2[2][BLOCK_SIZE / 16][16];

    const uint first_row = NUM_ROWS * (GID_0 + GDIM_0 * GID_2);

    // do NUM_ROWS at a time, unless there aren't enough remaining rows
    if (first_row + NUM_ROWS <= stride_d) {
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
            sccache1,
            sccache2,
            first_row, 
            NUM_ROWS);
    } else {
        if (first_row >= stride_d) {
            return;
        }
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
            sccache1,
            sccache2,
            first_row, 
            stride_d - first_row);
    }
} 

)";

const char g_kernelCodeMmv_Q3_K[] = R"(
inline void calc_superblock(
        const global block_q3_K *data_a,
        const global B_TYPE *data_b,
        const uint batch_stride_b,
        local FLOAT_TYPE sccache[2][BLOCK_SIZE / 16][2][8],
        FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], 
        uint *cselp,
        const uint a_offset, 
        const uint b_offset, 
        const uint ix, 
        const uint itid8, 
        const uint v_im, 
        const uint v_im4, 
        const uint v_in, 
        const uint hm_m[4], 
        const uint q_offset, 
        const uint y_offset, 
        const uint s_shift, 
        const uint i, 
        const uint num_blocks_per_row, 
        const uint first_row, 
        const uint num_rows, 
        const bool all_threads) {

    CAST_CONST_PACKED16(block_q3_K, data_a);
    const global B_TYPE_V2 *data_b_v2 = (const global B_TYPE_V2 *)data_b;

    const uint y_idx = i * QUANT_K + y_offset;

    uint csel = *cselp;

    unroll_for (uint n = 0; n < num_rows; n++) {
        const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
        csel ^= 1;

        if (!all_threads) { 
            // when we don't have enough blocks to use all threads
            if (i < num_blocks_per_row)
                sccache[csel][ix][v_im][itid8] = 
                    (FLOAT_TYPE)((char)(((data_a[ib0 + i].scales[itid8] >> v_im4) & 0xF) | 
                        (((data_a[ib0 + i].scales[itid8 % 4 + 8] >> s_shift) & 3) << 4)) - 32);
            barrier(CLK_LOCAL_MEM_FENCE);
            if (i >= num_blocks_per_row) {
                continue;
            }
        }

        const uint hmk = 
            ~((uint)(data_a_packed16[ib0 + i].hmask[v_in]) | 
                ((uint)(data_a_packed16[ib0 + i].hmask[v_in + 8]) << 16));
        const float4 hmk_0 = convert_float4(as_uchar4(((hmk & hm_m[0]) >> (v_im4)) << 2));
        const float4 hmk_1 = convert_float4(as_uchar4(((hmk & hm_m[1]) >> (1 + v_im4)) << 2));
        const float4 hmk_2 = convert_float4(as_uchar4(((hmk & hm_m[2]) >> (2 + v_im4)) << 2));
        const float4 hmk_3 = convert_float4(as_uchar4(((hmk & hm_m[3]) >> (3 + v_im4)) << 2));

        // 0, 1, 16, 17
        uint qs_u32 = 
            (uint)data_a[ib0 + i].qs[q_offset] | 
                ((uint)data_a[ib0 + i].qs[q_offset + 1] << 8);
        qs_u32 |= 
           ((uint)data_a[ib0 + i].qs[q_offset + 16] | 
               ((uint)data_a[ib0 + i].qs[q_offset + 17] << 8)) << 16;
        const float4 qs_u32_0 = convert_float4(as_uchar4(qs_u32 & 0x03030303));
        const float4 qs_u32_2 = convert_float4(as_uchar4((qs_u32 >> 2) & 0x03030303));
        const float4 qs_u32_4 = convert_float4(as_uchar4((qs_u32 >> 4) & 0x03030303));
        const float4 qs_u32_6 = convert_float4(as_uchar4((qs_u32 >> 6) & 0x03030303));

        if (all_threads) {
            sccache[csel][ix][v_im][itid8] = 
                (FLOAT_TYPE)((char)(((data_a[ib0 + i].scales[itid8] >> v_im4) & 0xF) | 
                    (((data_a[ib0 + i].scales[itid8 % 4 + 8] >> s_shift) & 3) << 4)) - 32);
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const FLOAT_TYPE d = (FLOAT_TYPE)data_a[ib0 + i].d;

        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            float2 b0 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 0]);
            float2 b16 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 8]);
            float2 b32 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 16]);
            float2 b48 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 24]);
            float2 b64 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 32]);
            float2 b80 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 40]);
            float2 b96 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 48]);
            float2 b112 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y_idx) / 2 + 56]);

            FLOAT_TYPE sum = (FLOAT_TYPE)0;
            unroll_for (int l = 0; l < 2; l++) {
                sum = 
                    DOT8_ACC(
                        (FLOAT_TYPE)b0[l] * sccache[csel][ix][v_im][0], qs_u32_0[l] - hmk_0[l],
                        (FLOAT_TYPE)b16[l] * sccache[csel][ix][v_im][1], qs_u32_0[l+2] - hmk_0[l + 2],
                        (FLOAT_TYPE)b32[l] * sccache[csel][ix][v_im][2], qs_u32_2[l] - hmk_1[l],
                        (FLOAT_TYPE)b48[l] * sccache[csel][ix][v_im][3], qs_u32_2[l + 2] - hmk_1[l + 2],
                        (FLOAT_TYPE)b64[l] * sccache[csel][ix][v_im][4], qs_u32_4[l] - hmk_2[l],
                        (FLOAT_TYPE)b80[l] * sccache[csel][ix][v_im][5], qs_u32_4[l + 2] - hmk_2[l + 2],
                        (FLOAT_TYPE)b96[l] * sccache[csel][ix][v_im][6], qs_u32_6[l] - hmk_3[l],
                        (FLOAT_TYPE)b112[l] * sccache[csel][ix][v_im][7], qs_u32_6[l + 2] - hmk_3[l + 2], 
                        sum);
            }

            temp[j][n] = fma(d, sum, temp[j][n]);
        }
    }

    *cselp = csel;
}

inline void compute_outputs(
        const global block_q3_K *data_a,
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
        uint a_bcast1,
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
#endif
        local FLOAT_TYPE sccache[2][BLOCK_SIZE / 16][2][8],
        const uint first_row, 
        const uint num_rows) {

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

    const uint num_blocks_per_row = ncols / QUANT_K;

    // 16 threads are used to process each block
    const uint it_size = LDIM_0 / 16;
    const uint tid = LID_0;
    const uint itid = tid % 16;        // 0...15
    const uint ix = tid / 16;
    const uint itid8 = itid % 8;

    const uint v_im = itid / 8;        // 0 or 1. 0 computes 0..., 1 computes 128...
    const uint v_im4 = v_im * 4;
    const uint v_in = itid - 8 * v_im; // 0...7

    const uint m = 0x01010101 << (4 * v_im);
    uint hm_m[4];
    unroll_for (uint j = 0; j < 4; j++) {
        hm_m[j] = m << j;
    }

    const uint l0 = 2 * v_in;              // 0...15
    const uint q_offset = 32 * v_im + l0;
    const uint y_offset = 128 * v_im + l0;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS] = {{(FLOAT_TYPE)0}}; 

    uint csel = 0;

    const uint s_shift = v_im4 + 2 * (itid8 / 4);

    const uint nbr_par_th = num_blocks_per_row % it_size;
    const uint nbr_all_th = num_blocks_per_row - nbr_par_th;
    uint i0 = 0;
    unroll_for ( ; i0 < nbr_all_th; i0 += it_size) {
        calc_superblock(
            data_a,
            data_b,
            batch_stride_b,
            sccache,
            temp, 
            &csel,
            a_offset, 
            b_offset, 
            ix, 
            itid8, 
            v_im, 
            v_im4, 
            v_in, 
            hm_m, 
            q_offset, 
            y_offset, 
            s_shift, 
            i0 + ix, 
            num_blocks_per_row, 
            first_row, 
            num_rows, 
            true);
    }
    calc_superblock(
        data_a,
        data_b,
        batch_stride_b,
        sccache,
        temp, 
        &csel,
        a_offset, 
        b_offset, 
        ix, 
        itid8, 
        v_im, 
        v_im4, 
        v_in, 
        hm_m, 
        q_offset, 
        y_offset, 
        s_shift, 
        i0 + ix, 
        num_blocks_per_row, 
        first_row, 
        num_rows, 
        false);

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
kernel void mul_mat_vec(
        const global block_q3_K *data_a,
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
    data_d += D_BASE / sizeof(float);

#ifndef USE_SUBGROUP_ADD_NO_SHMEM
    local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

    local FLOAT_TYPE sccache[2][BLOCK_SIZE / 16][2][8]; 

    const uint first_row = NUM_ROWS * (GID_0 + GDIM_0 * GID_2);

    // do NUM_ROWS at a time, unless there aren't enough remaining rows
    if (first_row + NUM_ROWS <= stride_d) {
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
            sccache,
            first_row, 
            NUM_ROWS);
    } else {
        if (first_row >= stride_d) {
            return;
        }
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
            sccache,
            first_row, 
            stride_d - first_row);
    }
} 

)";

const char g_kernelCodeMmv_Q4_K[] = R"(
inline void calc_superblock(
        const global block_q4_K *data_a,
        const global B_TYPE *data_b,
        const uint batch_stride_b,
        FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], 
        const uint a_offset, 
        const uint b_offset, 
        const uint v_im, 
        const uint q_offset, 
        const uint y_offset, 
        const uint i, 
        const uint num_blocks_per_row, 
        const uint first_row, 
        const uint num_rows) {

    CAST_CONST_PACKED16(block_q4_K, data_a);
    CAST_CONST_PACKED32(block_q4_K, data_a);
    const global B_TYPE_V4 *data_b_v4 = (const global B_TYPE_V4 *)data_b;

    const uint y1_idx = i * QUANT_K + y_offset;
    const uint y2_idx = y1_idx + 128;

    unroll_for (uint n = 0; n < num_rows; n++) {
        const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
        const FLOAT_TYPE_V2 dm = CONVERT_FLOAT_TYPE_V2(data_a[ib0 + i].dm);

        const uint scale0_u32 = data_a_packed16[ib0 + i].scales[v_im];
        const uint scale4_u32 = data_a_packed16[ib0 + i].scales[v_im + 2];
        const uint scale8_u32 = data_a_packed16[ib0 + i].scales[v_im + 4];

        const uint scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
        const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0) >> 2;
        const float4 scale_0_4_l_f = 
            convert_float4(as_uchar4(scale_0_4_l & 0x3F3F3F3F));
        const float4 scale8_f = 
            convert_float4(as_uchar4((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0F) | scale_0_4_h));

        const FLOAT_TYPE sc0 = scale_0_4_l_f.x;
        const FLOAT_TYPE sc1 = scale_0_4_l_f.y;
        const FLOAT_TYPE sc2 = scale_0_4_l_f.z;
        const FLOAT_TYPE sc3 = scale_0_4_l_f.w;
        const FLOAT_TYPE sc4 = scale8_f.x;
        const FLOAT_TYPE sc5 = scale8_f.y;
        const FLOAT_TYPE sc6 = scale8_f.z;
        const FLOAT_TYPE sc7 = scale8_f.w;

        const uint qs0_u32 = data_a_packed32[ib0 + i].qs[q_offset / 4];
        const uint qs64_u32 = data_a_packed32[ib0 + i].qs[q_offset / 4 + 16];

        const uint qs0_u32_lo4 = qs0_u32 & 0x0F0F0F0F;
        const uint qs0_u32_hi4 = (qs0_u32 >> 4) & 0x0F0F0F0F;
        const uint qs64_u32_lo4 = qs64_u32 & 0x0F0F0F0F;
        const uint qs64_u32_hi4 = (qs64_u32 >> 4) & 0x0F0F0F0F;

        const float4 qs0_lo4 = convert_float4(as_uchar4(qs0_u32_lo4));
        const float4 qs64_lo4 = convert_float4(as_uchar4(qs64_u32_lo4));
        const float4 qs0_hi4 = convert_float4(as_uchar4(qs0_u32_hi4));
        const float4 qs64_hi4 = convert_float4(as_uchar4(qs64_u32_hi4));

        const FLOAT_TYPE q4_0 = qs0_lo4.x;
        const FLOAT_TYPE q4_1 = qs0_lo4.y;
        const FLOAT_TYPE q4_2 = qs0_lo4.z;
        const FLOAT_TYPE q4_3 = qs0_lo4.w;
        const FLOAT_TYPE q4_4 = qs0_hi4.x;
        const FLOAT_TYPE q4_5 = qs0_hi4.y;
        const FLOAT_TYPE q4_6 = qs0_hi4.z;
        const FLOAT_TYPE q4_7 = qs0_hi4.w;
        const FLOAT_TYPE q4_8 = qs64_lo4.x;
        const FLOAT_TYPE q4_9 = qs64_lo4.y;
        const FLOAT_TYPE q4_10 = qs64_lo4.z;
        const FLOAT_TYPE q4_11 = qs64_lo4.w;
        const FLOAT_TYPE q4_12 = qs64_hi4.x;
        const FLOAT_TYPE q4_13 = qs64_hi4.y;
        const FLOAT_TYPE q4_14 = qs64_hi4.z;
        const FLOAT_TYPE q4_15 = qs64_hi4.w;

        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            float4 by10 = convert_float4(data_b_v4[(j * batch_stride_b + b_offset + y1_idx) / 4]);
            float4 by132 = convert_float4(data_b_v4[(j * batch_stride_b + b_offset + y1_idx) / 4 + 8]);
            float4 by20 = convert_float4(data_b_v4[(j * batch_stride_b + b_offset + y2_idx) / 4]);
            float4 by232 = convert_float4(data_b_v4[(j * batch_stride_b + b_offset + y2_idx) / 4 + 8]);

            const FLOAT_TYPE sx = 
                DOT4(
                    (FLOAT_TYPE)by10.x, q4_0,  
                    (FLOAT_TYPE)by10.y, q4_1,  
                    (FLOAT_TYPE)by10.z, q4_2,  
                    (FLOAT_TYPE)by10.w, q4_3);
            const FLOAT_TYPE sy = 
                DOT4(
                    (FLOAT_TYPE)by132.x, q4_4,  
                    (FLOAT_TYPE)by132.y, q4_5,  
                    (FLOAT_TYPE)by132.z, q4_6,  
                    (FLOAT_TYPE)by132.w, q4_7);
            const FLOAT_TYPE sz = 
                DOT4(
                    (FLOAT_TYPE)by20.x, q4_8,  
                    (FLOAT_TYPE)by20.y, q4_9,
                    (FLOAT_TYPE)by20.z, q4_10, 
                    (FLOAT_TYPE)by20.w, q4_11);
            const FLOAT_TYPE sw = 
                DOT4(
                    (FLOAT_TYPE)by232.x, q4_12, 
                    (FLOAT_TYPE)by232.y, q4_13, 
                    (FLOAT_TYPE)by232.z, q4_14, 
                    (FLOAT_TYPE)by232.w, q4_15);

            const FLOAT_TYPE smin =
                DOT16(
                    (FLOAT_TYPE)by10.x, sc2, 
                    (FLOAT_TYPE)by132.x, sc3, 
                    (FLOAT_TYPE)by20.x, sc6, 
                    (FLOAT_TYPE)by232.x, sc7,
                    (FLOAT_TYPE)by10.y, sc2, 
                    (FLOAT_TYPE)by132.y, sc3, 
                    (FLOAT_TYPE)by20.y, sc6, 
                    (FLOAT_TYPE)by232.y, sc7,
                    (FLOAT_TYPE)by10.z, sc2, 
                    (FLOAT_TYPE)by132.z, sc3, 
                    (FLOAT_TYPE)by20.z, sc6, 
                    (FLOAT_TYPE)by232.z, sc7,
                    (FLOAT_TYPE)by10.w, sc2, 
                    (FLOAT_TYPE)by132.w, sc3, 
                    (FLOAT_TYPE)by20.w, sc6,     
                    (FLOAT_TYPE)by232.w, sc7);

            const FLOAT_TYPE sum = DOT4(sx, sc0, sy, sc1, sz, sc4, sw, sc5);

            temp[j][n] = fma(dm.x, sum, fma(-dm.y, smin, temp[j][n]));
        }
    }
}

inline void compute_outputs(
        const global block_q4_K *data_a,
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
        uint a_bcast1,
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
#endif
        const uint first_row, 
        const uint num_rows) {

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

    const uint num_blocks_per_row = ncols / QUANT_K;

    // 16 threads are used to process each block
    const uint it_size = LDIM_0 / 16;
    const uint tid = LID_0;
    const uint itid = tid % 16;           // 0...15
    const uint ix = tid / 16;

    const uint il = itid / 4;             // 0...3
    const uint ir = itid - 4 * il;        // 0...3
    const uint n =  4;

    const uint v_im = il / 2;             // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const uint v_in = il % 2;

    const uint l0 = n * (2 * ir + v_in);  // 0...15
    const uint q_offset = 32 * v_im + l0;
    const uint y_offset = 64 * v_im + l0;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS] = {{(FLOAT_TYPE)0}}; 

    unroll_for (uint i = ix; i < num_blocks_per_row; i += it_size) {
        calc_superblock(
            data_a,
            data_b,
            batch_stride_b,
            temp, 
            a_offset, 
            b_offset, 
            v_im, 
            q_offset, 
            y_offset, 
            i, 
            num_blocks_per_row, 
            first_row, 
            num_rows);
    }

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
kernel void mul_mat_vec(
        const global block_q4_K *data_a,
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
    data_d += D_BASE / sizeof(float);

#ifndef USE_SUBGROUP_ADD_NO_SHMEM
    local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

    const uint first_row = NUM_ROWS * (GID_0 + GDIM_0 * GID_2);

    // do NUM_ROWS at a time, unless there aren't enough remaining rows
    if (first_row + NUM_ROWS <= stride_d) {
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
            NUM_ROWS);
    } else {
        if (first_row >= stride_d) {
            return;
        }
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
            stride_d - first_row);
    }
} 

)";

const char g_kernelCodeMmv_Q5_K[] = R"(
inline void calc_superblock(
        const global block_q5_K *data_a,
        const global B_TYPE *data_b,
        const uint batch_stride_b,
        FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], 
        const uint a_offset, 
        const uint b_offset, 
        const uint v_im, 
        const uint l0, 
        const uint q_offset, 
        const uint y_offset, 
        const uint i, 
        const uint num_blocks_per_row, 
        const uint first_row, 
        const uint num_rows) {

    CAST_CONST_PACKED16(block_q5_K, data_a);
    const global B_TYPE_V2 *data_b_v2 = (const global B_TYPE_V2 *)data_b;

    const uint y1_idx = i * QUANT_K + y_offset;
    const uint y2_idx = y1_idx + 128;

    unroll_for (uint n = 0; n < num_rows; n++) {
        const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
        const FLOAT_TYPE_V2 dm = CONVERT_FLOAT_TYPE_V2(data_a[ib0 + i].dm);

        const uint scale0_u32 = data_a_packed16[ib0 + i].scales[v_im];
        const uint scale4_u32 = data_a_packed16[ib0 + i].scales[v_im + 2];
        const uint scale8_u32 = data_a_packed16[ib0 + i].scales[v_im + 4];

        const uint scale_0_4_l = (scale4_u32 << 16) | scale0_u32;
        const uint scale_0_4_h = (scale_0_4_l & 0xC0C0C0C0) >> 2;
        const float4 scale_0_4_l_f = 
            convert_float4(as_uchar4(scale_0_4_l & 0x3F3F3F3F));
        const float4 scale8_f = 
            convert_float4(as_uchar4((((scale8_u32 << 12) | scale8_u32) & 0x0F0F0F0F) | scale_0_4_h));

        const FLOAT_TYPE sc0 = scale_0_4_l_f.x;
        const FLOAT_TYPE sc1 = scale_0_4_l_f.y;
        const FLOAT_TYPE sc2 = scale_0_4_l_f.z;
        const FLOAT_TYPE sc3 = scale_0_4_l_f.w;
        const FLOAT_TYPE sc4 = scale8_f.x;
        const FLOAT_TYPE sc5 = scale8_f.y;
        const FLOAT_TYPE sc6 = scale8_f.z;
        const FLOAT_TYPE sc7 = scale8_f.w;

        const uint qs0_16_u32 = 
            (uint)data_a_packed16[ib0 + i].qs[q_offset / 2] | 
                ((uint)data_a_packed16[ib0 + i].qs[q_offset / 2 + 8] << 16);
        const uint qs64_80_u32 = 
            (uint)data_a_packed16[ib0 + i].qs[q_offset / 2 + 32] | 
                ((uint)data_a_packed16[ib0 + i].qs[q_offset / 2 + 40] << 16);

        uint qs0_16_u32_lo4 = qs0_16_u32 & 0x0F0F0F0F;
        uint qs0_16_u32_hi4 = (qs0_16_u32 >> 4) & 0x0F0F0F0F;
        uint qs64_80_u32_lo4 = qs64_80_u32 & 0x0F0F0F0F;
        uint qs64_80_u32_hi4 = (qs64_80_u32 >> 4) & 0x0F0F0F0F;

        const uint qh = 
            as_uint((ushort2)(
                data_a_packed16[ib0 + i].qh[l0 / 2], 
                data_a_packed16[ib0 + i].qh[l0 / 2 + 8]));

        const uint qs0_16_lo4_offset16 = ((qh >> (2 * v_im)) & 0x01010101) << 4;
        const uint qs0_16_hi4_offset16 = ((qh >> (2 * v_im)) & 0x02020202) << 3;
        const uint qs64_80_lo4_offset16 = ((qh >> (2 * v_im)) & 0x10101010);
        const uint qs64_80_hi4_offset16 = ((qh >> (2 * v_im)) & 0x20202020) >> 1;

        qs0_16_u32_lo4 += qs0_16_lo4_offset16;
        qs0_16_u32_hi4 += qs0_16_hi4_offset16;
        qs64_80_u32_lo4 += qs64_80_lo4_offset16;
        qs64_80_u32_hi4 += qs64_80_hi4_offset16;

        const float4 qs0_16_lo4 = convert_float4(as_uchar4(qs0_16_u32_lo4));
        const float4 qs64_80_lo4 = convert_float4(as_uchar4(qs64_80_u32_lo4));
        const float4 qs0_16_hi4 = convert_float4(as_uchar4(qs0_16_u32_hi4));
        const float4 qs64_80_hi4 = convert_float4(as_uchar4(qs64_80_u32_hi4));

        const FLOAT_TYPE q4_0 = qs0_16_lo4.x;
        const FLOAT_TYPE q4_1 = qs0_16_lo4.y;
        const FLOAT_TYPE q4_2 = qs0_16_lo4.z;
        const FLOAT_TYPE q4_3 = qs0_16_lo4.w;
        const FLOAT_TYPE q4_4 = qs0_16_hi4.x;
        const FLOAT_TYPE q4_5 = qs0_16_hi4.y;
        const FLOAT_TYPE q4_6 = qs0_16_hi4.z;
        const FLOAT_TYPE q4_7 = qs0_16_hi4.w;
        const FLOAT_TYPE q4_8 = qs64_80_lo4.x;
        const FLOAT_TYPE q4_9 = qs64_80_lo4.y;
        const FLOAT_TYPE q4_10 = qs64_80_lo4.z;
        const FLOAT_TYPE q4_11 = qs64_80_lo4.w;
        const FLOAT_TYPE q4_12 = qs64_80_hi4.x;
        const FLOAT_TYPE q4_13 = qs64_80_hi4.y;
        const FLOAT_TYPE q4_14 = qs64_80_hi4.z;
        const FLOAT_TYPE q4_15 = qs64_80_hi4.w;

        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            float2 by10 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y1_idx) / 2]);
            float2 by116 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y1_idx) / 2 +  8]);
            float2 by132 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y1_idx) / 2 + 16]);
            float2 by148 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y1_idx) / 2 + 24]);
            float2 by20 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y2_idx) / 2]);
            float2 by216 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y2_idx) / 2 +  8]);
            float2 by232 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y2_idx) / 2 + 16]);
            float2 by248 = convert_float2(data_b_v2[(j * batch_stride_b + b_offset + y2_idx) / 2 + 24]);

            const FLOAT_TYPE sx =
                DOT4(
                    (FLOAT_TYPE)by10.x, q4_0,
                    (FLOAT_TYPE)by10.y, q4_1,
                    (FLOAT_TYPE)by116.x, q4_2,
                    (FLOAT_TYPE)by116.y, q4_3);
            const FLOAT_TYPE sy =
                DOT4(
                    (FLOAT_TYPE)by132.x, q4_4,
                    (FLOAT_TYPE)by132.y, q4_5,
                    (FLOAT_TYPE)by148.x, q4_6,
                    (FLOAT_TYPE)by148.y, q4_7);
            const FLOAT_TYPE sz =
                DOT4(
                    (FLOAT_TYPE)by20.x, q4_8,
                    (FLOAT_TYPE)by20.y, q4_9,
                    (FLOAT_TYPE)by216.x, q4_10,
                    (FLOAT_TYPE)by216.y, q4_11);
            const FLOAT_TYPE sw =
                DOT4(
                    (FLOAT_TYPE)by232.x, q4_12,
                    (FLOAT_TYPE)by232.y, q4_13,
                    (FLOAT_TYPE)by248.x, q4_14,
                    (FLOAT_TYPE)by248.y, q4_15);

            const FLOAT_TYPE smin =
                DOT4(
                    (FLOAT_TYPE)by10.x + 
                        (FLOAT_TYPE)by10.y + 
                        (FLOAT_TYPE)by116.x + 
                        (FLOAT_TYPE)by116.y, 
                    sc2,
                    (FLOAT_TYPE)by132.x + 
                        (FLOAT_TYPE)by132.y + 
                        (FLOAT_TYPE)by148.x + 
                        (FLOAT_TYPE)by148.y, 
                    sc3,
                    (FLOAT_TYPE)by20.x + 
                        (FLOAT_TYPE)by20.y + 
                        (FLOAT_TYPE)by216.x + 
                        (FLOAT_TYPE)by216.y, 
                    sc6,
                    (FLOAT_TYPE)by232.x + 
                        (FLOAT_TYPE)by232.y + 
                        (FLOAT_TYPE)by248.x + 
                        (FLOAT_TYPE)by248.y,
                    sc7);

            FLOAT_TYPE sum = DOT4(sx, sc0, sy, sc1, sz, sc4, sw, sc5);

            temp[j][n] = fma(dm.x, sum, fma(-dm.y, smin, temp[j][n]));
        }
    }
}

inline void compute_outputs(
        const global block_q5_K *data_a,
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
        uint a_bcast1,
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
#endif
        const uint first_row, 
        const uint num_rows) {

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

    const uint num_blocks_per_row = ncols / QUANT_K;

    // 16 threads are used to process each block
    const uint it_size = LDIM_0 / 16;
    const uint tid = LID_0;
    const uint itid = tid % 16;           // 0...15
    const uint ix = tid / 16;

    const uint il = itid / 4;             // 0...3
    const uint ir = itid - 4 * il;        // 0...3

    const uint v_im = il / 2;             // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const uint v_in = il % 2;

    const uint l0 = 4 * ir + 2 * v_in;    // 0...15
    const uint q_offset = 32 * v_im + l0;
    const uint y_offset = 64 * v_im + l0;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS] = {{(FLOAT_TYPE)0}}; 

    unroll_for (uint i = ix; i < num_blocks_per_row; i += it_size) {
        calc_superblock(
            data_a,
            data_b,
            batch_stride_b,
            temp, 
            a_offset, 
            b_offset, 
            v_im, 
            l0, 
            q_offset, 
            y_offset, 
            i, 
            num_blocks_per_row, 
            first_row, 
            num_rows);
    }

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
kernel void mul_mat_vec(
        const global block_q5_K *data_a,
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
    data_d += D_BASE / sizeof(float);

#ifndef USE_SUBGROUP_ADD_NO_SHMEM
    local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

    const uint first_row = NUM_ROWS * (GID_0 + GDIM_0 * GID_2);

    // do NUM_ROWS at a time, unless there aren't enough remaining rows
    if (first_row + NUM_ROWS <= stride_d) {
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
            NUM_ROWS);
    } else {
        if (first_row >= stride_d) {
            return;
        }
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
            stride_d - first_row);
    }
} 

)";

const char g_kernelCodeMmv_Q6_K[] = R"(
inline void calc_superblock(
        const global block_q6_K *data_a,
        const global B_TYPE *data_b,
        const uint batch_stride_b,
        local FLOAT_TYPE sccache[2][BLOCK_SIZE / 16][16],
        FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], 
        uint *cselp,
        const uint a_offset, 
        const uint b_offset, 
        const uint itid, 
        const uint ix, 
        const uint ql_offset, 
        const uint qh_offset, 
        const uint s_offset, 
        const uint y_offset, 
        const uint i, 
        const uint num_blocks_per_row, 
        const uint first_row, 
        const uint num_rows, 
        const bool all_threads) {

    CAST_CONST_PACKED16(block_q6_K, data_a);
    const global B_TYPE_V4 *data_b_v4 = (const global B_TYPE_V4 *)data_b;

    const uint y_idx = i * QUANT_K + y_offset;

    uint csel = *cselp;

    unroll_for (uint n = 0; n < num_rows; n++) {
        const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;
        csel ^= 1;

        if (!all_threads) { 
            // when we don't have enough blocks to use all threads
            if (i < num_blocks_per_row) {
                sccache[csel][ix][itid] = (FLOAT_TYPE)data_a[ib0 + i].scales[itid];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (i >= num_blocks_per_row) {
                continue;
            }
        }

        const uint ql0_u32 = 
            (uint)data_a_packed16[ib0 + i].ql[ql_offset / 2] | 
                ((uint)data_a_packed16[ib0 + i].ql[ql_offset / 2 + 1] << 16);
        const uint ql32_u32 = 
            (uint)data_a_packed16[ib0 + i].ql[ql_offset / 2 + 16] | 
                ((uint)data_a_packed16[ib0 + i].ql[ql_offset / 2 + 17] << 16);

        const uint ql0_u32_lo4 = ql0_u32 & 0x0F0F0F0F;
        const uint ql0_u32_hi4 = (ql0_u32 >> 4) & 0x0F0F0F0F;
        const uint ql32_u32_lo4 = ql32_u32 & 0x0F0F0F0F;
        const uint ql32_u32_hi4 = (ql32_u32 >> 4) & 0x0F0F0F0F;

        const uint qh_u32 = 
             (uint)data_a_packed16[ib0 + i].qh[qh_offset / 2] | 
                 ((uint)data_a_packed16[ib0 + i].qh[qh_offset / 2 + 1] << 16);

        const uint qh0_u32 = (qh_u32 & 0x03030303) << 4;
        const uint qh2_u32 = (qh_u32 & 0x0C0C0C0C) << 2;
        const uint qh4_u32 = (qh_u32 & 0x30303030);
        const uint qh6_u32 = (qh_u32 & 0xC0C0C0C0) >> 2;

        const uint q0_u32 = ql0_u32_lo4 | qh0_u32;
        const uint q1_u32 = ql32_u32_lo4 | qh2_u32;
        const uint q2_u32 = ql0_u32_hi4 | qh4_u32;
        const uint q3_u32 = ql32_u32_hi4 | qh6_u32;

        const float4 q0 = convert_float4(as_uchar4(q0_u32)) - 32;
        const float4 q1 = convert_float4(as_uchar4(q1_u32)) - 32;
        const float4 q2 = convert_float4(as_uchar4(q2_u32)) - 32;
        const float4 q3 = convert_float4(as_uchar4(q3_u32)) - 32;

        if (all_threads) {
            sccache[csel][ix][itid] = (FLOAT_TYPE)data_a[ib0 + i].scales[itid];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        const FLOAT_TYPE d = (FLOAT_TYPE)data_a[ib0 + i].d;

        unroll_for (uint j = 0; j < NUM_COLS; j++) {
            float4 by0 = convert_float4(data_b_v4[(j * batch_stride_b + b_offset + y_idx) / 4]);
            float4 by32 = convert_float4(data_b_v4[(j * batch_stride_b + b_offset + y_idx) / 4 +  8]);
            float4 by64 = convert_float4(data_b_v4[(j * batch_stride_b + b_offset + y_idx) / 4 + 16]);
            float4 by96 = convert_float4(data_b_v4[(j * batch_stride_b + b_offset + y_idx) / 4 + 24]);

            FLOAT_TYPE sum[4] = {(FLOAT_TYPE)0};
            unroll_for (uint l = 0; l < 4; l++) {
                sum[0] = fma((FLOAT_TYPE)by0[l], q0[l], sum[0]);
                sum[1] = fma((FLOAT_TYPE)by32[l], q1[l], sum[1]);
                sum[2] = fma((FLOAT_TYPE)by64[l], q2[l], sum[2]);
                sum[3] = fma((FLOAT_TYPE)by96[l], q3[l], sum[3]);
            }

            FLOAT_TYPE sum4 =
                DOT4(
                    sum[0], sccache[csel][ix][s_offset], 
                    sum[1], sccache[csel][ix][s_offset + 2], 
                    sum[2], sccache[csel][ix][s_offset + 4], 
                    sum[3], sccache[csel][ix][s_offset + 6]); 

            temp[j][n] = fma(sum4, d, temp[j][n]);
        }
    }

    *cselp = csel;
}

inline void compute_outputs(
        const global block_q6_K *data_a,
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
        uint a_bcast1,
#ifndef USE_SUBGROUP_ADD_NO_SHMEM
        local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE],
#endif
        local FLOAT_TYPE sccache[2][BLOCK_SIZE / 16][16],
        const uint first_row, 
        const uint num_rows) {

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

    const uint num_blocks_per_row = ncols / QUANT_K;

    // 16 threads are used to process each block
    const uint it_size = LDIM_0 / 16;
    const uint tid = LID_0;
    const uint itid = tid % 16;  // 0...15
    const uint ix = tid / 16;

    const uint v_im = itid / 8;        // 0 or 1. 0 computes 0..., 1 computes 128...
    const uint v_in = itid - 8 * v_im; // 0...7

    const uint l0 = 4 * v_in;          // 0, 4, 8, ..., 28
    const uint is = v_in / 4;

    const uint ql_offset = 64 * v_im + l0;
    const uint qh_offset = 32 * v_im + l0;
    const uint s_offset = 8 * v_im + is;
    const uint y_offset = 128 * v_im + l0;

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS] = {{(FLOAT_TYPE)0}};
    
    uint csel = 0;
 
    const uint nbr_par_th = num_blocks_per_row % it_size;
    const uint nbr_all_th = num_blocks_per_row - nbr_par_th;
    uint i0 = 0;
    unroll_for ( ; i0 < nbr_all_th; i0 += it_size) {
        calc_superblock(
            data_a,
            data_b,
            batch_stride_b,
            sccache,
            temp, 
            &csel,
            a_offset, 
            b_offset, 
            itid, 
            ix, 
            ql_offset, 
            qh_offset, 
            s_offset, 
            y_offset, 
            i0 + ix, 
            num_blocks_per_row, 
            first_row, 
            num_rows, 
            true);
    }
    calc_superblock(
        data_a,
        data_b,
        batch_stride_b,
        sccache,
        temp, 
        &csel,
        a_offset, 
        b_offset, 
        itid, 
        ix, 
        ql_offset, 
        qh_offset, 
        s_offset, 
        y_offset, 
        i0 + ix, 
        num_blocks_per_row, 
        first_row, 
        num_rows, 
        false);

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
kernel void mul_mat_vec(
        const global block_q6_K *data_a,
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
    data_d += D_BASE / sizeof(float);

#ifndef USE_SUBGROUP_ADD_NO_SHMEM
    local FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

    local FLOAT_TYPE sccache[2][BLOCK_SIZE / 16][16];

    const uint first_row = NUM_ROWS * (GID_0 + GDIM_0 * GID_2);

    // do NUM_ROWS at a time, unless there aren't enough remaining rows
    if (first_row + NUM_ROWS <= stride_d) {
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
            sccache,
            first_row, 
            NUM_ROWS);
    } else {
        if (first_row >= stride_d) {
            return;
        }
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
            sccache,
            first_row, 
            stride_d - first_row);
    }
} 

)";

} // namespace

//
//    MulMatVecV2Common
//

const char *MulMatVecV2CommonCode() {
    return g_codeMmvCommon;
}

//
//    MulMatVecV2Impl
//

const char *MulMatVecV2Impl_Q4_0_Code() {
    return g_codeMmvImpl_Q4_0;
}

const char *MulMatVecV2Impl_Q4_1_Code() {
    return g_codeMmvImpl_Q4_1;
}

const char *MulMatVecV2Impl_Q5_0_Code() {
    return g_codeMmvImpl_Q5_0;
}

const char *MulMatVecV2Impl_Q5_1_Code() {
    return g_codeMmvImpl_Q5_1;
}

const char *MulMatVecV2Impl_Q8_0_Code() {
    return g_codeMmvImpl_Q8_0;
}

const char *MulMatVecV2Impl_Q2_K_Code() {
    return g_codeMmvImpl_Q2_K;
}

const char *MulMatVecV2Impl_Q3_K_Code() {
    return g_codeMmvImpl_Q3_K;
}

const char *MulMatVecV2Impl_Q4_K_Code() {
    return g_codeMmvImpl_Q4_K;
}

const char *MulMatVecV2Impl_Q5_K_Code() {
    return g_codeMmvImpl_Q5_K;
}

const char *MulMatVecV2Impl_Q6_K_Code() {
    return g_codeMmvImpl_Q6_K;
}

const char *MulMatVecV2Impl_Mxfp4_Code() {
    return g_codeMmvImpl_Mxfp4;
}

//
//    MulMatVecV2
//

const char *MulMatVecV2KernelCode() {
    return g_kernelCodeMmv;
}

const char *MulMatVecV2_Q2_K_KernelCode() {
    return g_kernelCodeMmv_Q2_K;
}

const char *MulMatVecV2_Q3_K_KernelCode() {
    return g_kernelCodeMmv_Q3_K;
}

const char *MulMatVecV2_Q4_K_KernelCode() {
    return g_kernelCodeMmv_Q4_K;
}

const char *MulMatVecV2_Q5_K_KernelCode() {
    return g_kernelCodeMmv_Q5_K;
}

const char *MulMatVecV2_Q6_K_KernelCode() {
    return g_kernelCodeMmv_Q6_K;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

