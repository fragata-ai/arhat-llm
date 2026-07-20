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

const char g_kernelCodeCommon[] = R"(
float rope_yarn_ramp(float low, float high, int i3) {
    const float y = (i3 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
float2 rope_yarn(
        float theta_extrap, 
        float freq_scale, 
        float corr_low, 
        float corr_high,
        int i3, 
        float ext_factor, 
        float mscale) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_low, corr_high, i3) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return (float2)(cos(theta) * mscale, sin(theta) * mscale);
}

)";

const char g_kernelCodeRopeNorm[] = R"(
kernel void rope_simple(
        const global DATA_T *src0,
        const global int *src1,
        const global float *src2,
        global DATA_T *dst,
        int n_dims,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float corr_low,
        float corr_high
        SHAPE_INFO_ARGS) {

    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    if (src2 != NULL) {
        src2 += SRC2_BASE;
    }
    dst += DST_BASE;

    int i0 = GID_2;
    int i1 = GID_1;
    int i2 = GID_0;

    float theta_base = (float)src1[i1];
    float inv_ndims = -1.0f / n_dims;

    for (int i3 = 2 * LID_0; i3 < DST_D3; i3 += 2 * LDIM_0) {
        if (i3 < n_dims) {
            int ic = i3 / 2;
            float theta = theta_base * pow(freq_base, inv_ndims * i3);
            float freq_factor = (src2 != NULL) ? src2[ic] : 1.0f;
            float2 cos_sin_theta = 
                rope_yarn(
                    theta / freq_factor, 
                    freq_scale, 
                    corr_low, 
                    corr_high,
                    i3, 
                    ext_factor, 
                    attn_factor);
            const global DATA_T *src_ptr = src0 + SRC0_OFF(i0, i1, i2, i3);
            global DATA_T *dst_ptr = dst + DST_OFF(i0, i1, i2, i3);
            float x0 = TO_FLOAT(src_ptr[0]);
            float x1 = TO_FLOAT(src_ptr[1]);
            dst_ptr[0] = TO_DATA_T(x0 * cos_sin_theta.s0 - x1 * cos_sin_theta.s1);
            dst_ptr[1] = TO_DATA_T(x0 * cos_sin_theta.s1 + x1 * cos_sin_theta.s0);
        } else {
            const global DATA_T *src_ptr = src0 + SRC0_OFF(i0, i1, i2, i3);
            global DATA_T *dst_ptr = dst + DST_OFF(i0, i1, i2, i3);
            dst_ptr[0] = src_ptr[0];
            dst_ptr[1] = src_ptr[1];
        }
    }
}

)";

const char g_kernelCodeRopeNeox[] = R"(
kernel void rope_simple(
        const global DATA_T *src0,
        const global int *src1,
        const global float *src2,
        global DATA_T *dst,
        int n_dims,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float corr_low,
        float corr_high
        SHAPE_INFO_ARGS) {

    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    if (src2 != NULL) {
        src2 += SRC2_BASE;
    }
    dst += DST_BASE;

    int i0 = GID_2;
    int i1 = GID_1;
    int i2 = GID_0;

    float theta_base = (float)src1[i1];
    float inv_ndims = -1.0f / n_dims;

    for (int i3 = 2 * LID_0; i3 < DST_D3; i3 += 2 * LDIM_0) {
        if (i3 < n_dims) {
            int ic = i3 / 2;
            const float theta = theta_base * pow(freq_base, inv_ndims * i3);
            const float freq_factor = (src2 != NULL) ? src2[ic] : 1.0f;
            float2 cos_sin_theta = 
                rope_yarn(
                    theta / freq_factor, 
                    freq_scale, 
                    corr_low, 
                    corr_high,
                    i3, 
                    ext_factor, 
                    attn_factor);
            const global DATA_T *src_ptr = src0 + SRC0_OFF(i0, i1, i2, ic);
            global DATA_T *dst_ptr = dst + DST_OFF(i0, i1, i2, ic);
            const float x0 = TO_FLOAT(src_ptr[0]);
            const float x1 = TO_FLOAT(src_ptr[n_dims / 2]);
            dst_ptr[0] = TO_DATA_T(x0 * cos_sin_theta.s0 - x1 * cos_sin_theta.s1);
            dst_ptr[n_dims / 2] = TO_DATA_T(x0 * cos_sin_theta.s1 + x1 * cos_sin_theta.s0);
        } else {
            const global DATA_T *src_ptr = src0 + SRC0_OFF(i0, i1, i2, i3);
            global DATA_T *dst_ptr = dst + DST_OFF(i0, i1, i2, i3);
            dst_ptr[0] = src_ptr[0];
            dst_ptr[1] = src_ptr[1];
        }
    }
}

)";

const char g_kernelCodeRopeMulti[] = R"(
kernel void rope_simple(
        const global DATA_T *src0,
        const global int *src1,
        const global float *src2,
        global DATA_T *dst,
        int n_dims,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float corr_low,
        float corr_high,
        int sect0,
        int sect1,
        int sect2,
        int sect3
        SHAPE_INFO_ARGS) {

    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    if (src2 != NULL) {
        src2 += SRC2_BASE;
    }
    dst += DST_BASE;

    int i0 = GID_2;
    int i1 = GID_1;
    int i2 = GID_0;

    const int sect_dims = sect0 + sect1 + sect2 + sect3;
    const int sec_w = sect1 + sect0;

    float inv_ndims = -1.0f / n_dims;

    for (int i3 = 2 * LID_0; i3 < DST_D3; i3 += 2 * LDIM_0) {
        if (i3 < n_dims) {
            int ic = i3 / 2;
            const int sector = (i3 / 2) % sect_dims;
            float theta_base = 0.0f;
            if (IS_IMROPE) {
                if (sector % 3 == 1 && sector < 3 * sect1) { // h
                    theta_base = (float)src1[i1 + DST_D1 * 1];
                } else if (sector % 3 == 2 && sector < 3 * sect2) { // w
                    theta_base = (float)src1[i1 + DST_D1 * 2];
                } else if (sector % 3 == 0 && sector < 3 * sect0) { // t
                    theta_base = (float)src1[i1 + DST_D1 * 0];
                } else { // e
                    theta_base = (float)src1[i1 + DST_D1 * 3];
                }
            } else {
                if (sector < sect0) {
                    theta_base = src1[i1];
                } else if (sector >= sect0 && sector < sec_w) {
                    theta_base = src1[i1 + DST_D1 * 1];
                } else if (sector >= sec_w && sector < sec_w + sect2) {
                    theta_base = src1[i1 + DST_D1 * 2];
                } else if (sector >= sec_w + sect2) {
                    theta_base = src1[i1 + DST_D1 * 3];
                }
            }
            const float theta = theta_base * pow(freq_base, inv_ndims * i3);
            const float freq_factor = (src2 != NULL) ? src2[ic] : 1.0f;
            float2 cos_sin_theta = 
                rope_yarn(
                    theta / freq_factor, 
                    freq_scale, 
                    corr_low, 
                    corr_high,
                    i3, 
                    ext_factor, 
                    attn_factor);
            const global DATA_T *src_ptr = src0 + SRC0_OFF(i0, i1, i2, ic);
            global DATA_T *dst_ptr = dst + DST_OFF(i0, i1, i2, ic);
            const float x0 = TO_FLOAT(src_ptr[0]);
            const float x1 = TO_FLOAT(src_ptr[n_dims / 2]);
            dst_ptr[0] = TO_DATA_T(x0 * cos_sin_theta.s0 - x1 * cos_sin_theta.s1);
            dst_ptr[n_dims / 2] = TO_DATA_T(x0 * cos_sin_theta.s1 + x1 * cos_sin_theta.s0);
        } else {
            const global DATA_T *src_ptr = src0 + SRC0_OFF(i0, i1, i2, i3);
            global DATA_T *dst_ptr = dst + DST_OFF(i0, i1, i2, i3);
            dst_ptr[0] = src_ptr[0];
            dst_ptr[1] = src_ptr[1];
        }
    }
}

)";

const char g_kernelCodeRopeVision[] = R"(
kernel void rope_simple(
        global DATA_T *src0,
        global int *src1,
        global float *src2,
        global DATA_T *dst,
        int n_dims,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float corr_low,
        float corr_high,
        int sect0,
        int sect1
        SHAPE_INFO_ARGS) {

    src0 += SRC0_BASE;
    src1 += SRC1_BASE;
    if (src2 != NULL) {
        src2 += SRC2_BASE;
    }
    dst += DST_BASE;

    int i0 = GID_2;
    int i1 = GID_1;
    int i2 = GID_0;

    const int sect_dims = sect0 + sect1;
    const int sec_w = sect1 + sect0;

    float inv_ndims = -1.0f / n_dims;

    for (int i3 = 2 * LID_0; i3 < DST_D3; i3 += 2 * LDIM_0) {
        int ic = i3 / 2;
        const int sector = (i3 / 2) % sect_dims;
        float theta_base = 0.0f;
        if (sector < sect0) {
            const int p = sector;
            theta_base = src1[i1] * pow(freq_base, inv_ndims * 2.0f * p);
        } else if (sector >= sect0 && sector < sec_w) {
            const int p = sector - sect0;
            theta_base = src1[i1 + DST_D1] * pow(freq_base, inv_ndims * 2.0f * p);
        }
        const float freq_factor = (src2 != NULL) ? src2[ic] : 1.0f;
        float2 cos_sin_theta = 
            rope_yarn(
                theta_base / freq_factor, 
                freq_scale, 
                corr_low, 
                corr_high,
                i3, 
                ext_factor, 
                attn_factor);
        const global DATA_T *src_ptr = src0 + SRC0_OFF(i0, i1, i2, ic);
        global DATA_T *dst_ptr = dst + DST_OFF(i0, i1, i2, ic);
        const float x0 = TO_FLOAT(src_ptr[0]);
        const float x1 = TO_FLOAT(src_ptr[n_dims]);
        dst_ptr[0] = TO_DATA_T(x0 * cos_sin_theta.s0 - x1 * cos_sin_theta.s1);
        dst_ptr[n_dims] = TO_DATA_T(x0 * cos_sin_theta.s1 + x1 * cos_sin_theta.s0);
    }
}

)";

} // namespace

const char *RopeSimpleCommonCode() {
    return g_kernelCodeCommon;
}

const char *RopeSimpleNormKernelCode() {
    return g_kernelCodeRopeNorm;
}

const char *RopeSimpleNeoxKernelCode() {
    return g_kernelCodeRopeNeox;
}

const char *RopeSimpleMultiKernelCode() {
    return g_kernelCodeRopeMulti;
}

const char *RopeSimpleVisionKernelCode() {
    return g_kernelCodeRopeVision;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

