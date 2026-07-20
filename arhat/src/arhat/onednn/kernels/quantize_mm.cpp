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

const char g_kernelCodeQuantizeMm_Mfxp4[] = R"(
#define QK_MXFP4 32

// d4 - 8 E8M0 scales (1 per 32 values), 2 packed per uint32: d4[0]={s0,s1}, d4[1]={s2,s3}, etc.
// qs - 256 FP4 values packed as 4-bit pairs (2 per byte), 8 blocks of 32 values

struct block_fp4_mmq {
    uint d4[4];
    char qs[4 * 32];
}; 

#define BLOCK_FP4_MMQ_SIZE (8 * QK_MXFP4) // 256 values

// FP4 E2M1: max exponent (unbiased) is 2
#define FP4_E2M1_EMAX 2

inline uchar compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    const float e = log2(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = convert_int_rte(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return convert_uchar(biased);
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

// Positive LUT
constant float pos_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

inline uchar float_to_fp4_e2m1(float x, float e) {
    const uchar sign_bit = (x < 0.0f) << 3;
    float ax = fabs(x) * e;

    int best_i = 0;
    float best_err = fabs(ax - pos_lut[0]);

    unroll_for (int i = 1; i < 8; ++i) {
        const float err = fabs(ax - pos_lut[i]);
        if (err < best_err) {
            best_err = err;
            best_i = i;
        }
    }

    return convert_uchar(best_i | sign_bit);
}

// Each subgroup processes 2 blocks of 32 = 64 values

#define VALS_PER_SCALE 32
#define VALS_PER_SG (2 * VALS_PER_SCALE)

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0 - a31 is represented as a0a16, a1a17 ... a15a31

__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void quantize_mm(
        const global float *x,
        const global int *ids,
        global char *vy
        SHAPE_INFO_ARGS) {

    x += SRC0_BASE;
    if (ids != NULL) {
        ids += SRC1_BASE;
    }
    vy += DST_BASE;

    const int sg_id = LID_1;
    const int lane_id_32 = LID_0;

    const int nsgs = LDIM_1;

    const long sg_start_offset = (GID_1 * nsgs + sg_id) * VALS_PER_SG;

    if (sg_start_offset >= DST_D3) {
        return;
    }

    const long i2 = GID_0;
    const long i1 = GID_2 % DST_D1;
    const long i0 = GID_2 / DST_D1;

    const long iid = (ids != NULL) ? ids[i2] : i2;

    global block_fp4_mmq *y = (global block_fp4_mmq *)vy;

    const long ib0 = GID_2 * ((long)DST_D2 * (DST_D3 / BLOCK_FP4_MMQ_SIZE));
    const long ib = ib0 + (sg_start_offset / BLOCK_FP4_MMQ_SIZE) * DST_D2 + GID_0;
    const long quad_idx_in_block = (sg_start_offset % BLOCK_FP4_MMQ_SIZE) / VALS_PER_SG;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;
    global char2 *yqs2 = (global char2 *)y[ib].qs;

    const long base_pos = i0 * SRC0_S0 + i1 * SRC0_S1 + iid * SRC0_S2;

    uchar scales[2];

    unroll_for (int b = 0; b < 2; b++) {
        const long i3 = sg_start_offset + b * VALS_PER_SCALE + lane_id_32;
        const float xi = (i3 < SRC0_D3) ? x[base_pos + i3] : 0.0f;

        float amax = fabs(xi);
        unroll_for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmax(amax, sub_group_shuffle_xor(amax, mask));
        }

        const uchar e = compute_e8m0_scale(amax);
        scales[b] = e;
        // ACHTUNG: No direct OpenCL match for CUDA __frcp_rn
        const float inv_s = (amax == 0.0f) ? 0.0f : 1.0f / e8m0_to_fp32(e);

        // Fallback: manual FP4 conversion using LUT
        const uchar q_val = float_to_fp4_e2m1(xi, inv_s);

        const uchar q_lo_0 = sub_group_shuffle_xor(q_val, base);
        const uchar q_lo_1 = sub_group_shuffle_xor(q_val, base + 1);
        const uchar q_hi_0 = sub_group_shuffle_xor(q_val, base + 16);
        const uchar q_hi_1 = sub_group_shuffle_xor(q_val, base + 17);

        if (lane_in_group == 0) {
            char2 q;
            q.x = (q_hi_0 << 4) | q_lo_0;
            q.y = (q_hi_1 << 4) | q_lo_1;
            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
    }

    if (lane_id_32 == 0) {
        // Store 2 scales packed into 1 uint32
        y[ib].d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
    }
}

)";

const char g_kernelCodeQuantizeMm_Q8_1[] = R"(
// The y float data is converted to a data layout that can simply be copied to shared memory
//     as a contiguous block.
// The y float data is first grouped as blocks of 128 values.
// These blocks are then treated as individual data values and transposed.
//
// To avoid shared memory bank conflicts each block is padded with 16 bytes.
// This padding is also used to store block scales/partial sums.
// The scales multiplied with the quantized data are equal to the unquantized values.
// The partial sums are obtained by summing up a subgroup of the contained values
//     (prior to quantization) and are only needed for performance reasons.
//
// The exact data stored depends on the x data type.
//
//     float d4[4]    1 32 bit scale per 32 values, stored as d0, d1, d2, d3
//     half2 ds4[4]   1 16 bit scale + 1 16 bit partial sum per 32 values,
//                        stored as d0, s0, d1, s1, d2, s2, d3, s3
//     half d2s6[8]   1 16 bit scale per 64 values + 
//                        1 16 bit partial sum per 16 values for the first 96 values,
//                        stored as d0, d1, s1, s2, s3, s4, s5

#define QK8_1 32

typedef struct {
    union {
        float d4[4];
        half2 ds4[4];
        half d2s6[8];
    };
    char qs[4 * QK8_1]; // 128 values quantized to 8 bit each
} block_q8_1_mmq; 

#define MMQ_Q8_1_DS_LAYOUT_D4 0
#define MMQ_Q8_1_DS_LAYOUT_DS4 1
#define MMQ_Q8_1_DS_LAYOUT_D2S6 2

__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void quantize_mm(
        const global float *x, 
        const global int *ids, 
        global char *vy
        SHAPE_INFO_ARGS) {

    x += SRC0_BASE;
    if (ids != NULL) {
        ids += SRC1_BASE;
    }
    vy += DST_BASE;

    const long i3 = ((long)LDIM_0 * GID_1 + LID_0) * 4;

    if (i3 >= DST_D3) {
        return;
    }

    const long i2 = GID_0;
    const long i1 = GID_2 % DST_D1;
    const long i0 = GID_2 / DST_D1;

    const long iid = (ids != NULL) ? ids[i2] : i2;

    const global float4 *x4 = (const global float4 *)x;

    global block_q8_1_mmq *y = (global block_q8_1_mmq *)vy;

    const long ib0 = GID_2 * ((long)GDIM_0 * GDIM_1 * LDIM_0 / QK8_1); // first block of channel
    const long ib = ib0 + (i3 / (4 * QK8_1)) * DST_D2 + GID_0;         // block index in channel
    const long iqs = i3 % (4 * QK8_1);                                 // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = 
        (i3 < SRC0_D3) ? 
            x4[(i0 * SRC0_S0 + i1 * SRC0_S1 + iid * SRC0_S2 + i3) / 4] : 
            (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabs(xi.x);
    amax = fmax(amax, fabs(xi.y));
    amax = fmax(amax, fabs(xi.z));
    amax = fmax(amax, fabs(xi.w));

    // Exchange max. abs. value between VALS_PER_SCALE / 4 threads
    unroll_for (int offset = VALS_PER_SCALE / 8; offset > 0; offset >>= 1) {
        amax = fmax(amax, sub_group_shuffle_xor(amax, offset));
    }

    float sum;
    if (DS_LAYOUT != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across VALS_PER_SUM / 4 threads.
        unroll_for (int offset = VALS_PER_SUM / 8; offset > 0; offset >>= 1) {
            sum += sub_group_shuffle_xor(sum, offset);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q = convert_char4(round(xi * d_inv));

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth
    global char4 *yqs4 = (global char4 *)y[ib].qs;
    yqs4[iqs / 4] = q;

    if (DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }
        y[ib].d2s6[2 + iqs / 16] = sum;
        if (iqs % 64 != 0) {
            return;
        }
        const float d = 1.0f / d_inv;
        y[ib].d2s6[iqs / 64] = d;
        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (DS_LAYOUT == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs / 32] = (half2)(d, sum);
    } else {
        y[ib].d4[iqs / 32]  = d;
    }
}

)";

} // namespace

const char *QuantizeMm_Mfxp4_KernelCode() {
    return g_kernelCodeQuantizeMm_Mfxp4;
}

const char *QuantizeMm_Q8_1_KernelCode() {
    return g_kernelCodeQuantizeMm_Q8_1;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

