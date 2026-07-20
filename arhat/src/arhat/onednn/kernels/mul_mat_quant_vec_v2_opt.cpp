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
//    MulMatQuantVecV2OptImpl
//

const char g_codeMmvqImpl_Q4_0[] = R"(
inline ushort3 load(const global ushort *data) {
    const uint tid = get_sub_group_local_id();
    ushort3 src;
    unroll_for (int i = 0; i < 2; i++) {
        src[i] = data[i * 16 + tid];
    }
    src[2] = (tid < 4) ? data[2 * 16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline ushort2 get_qs(const ushort3 src) {
    ushort2 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<2;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,10)<2;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,3)<2;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,12)<2;1,0>\n" 

        "mov (4) %0(1,0)<1> %1(0,2)<2;1,0>\n" 
        "mov (4) %0(1,4)<1> %1(0,11)<2;1,0>\n"
        "mov (4) %0(1,8)<1> %1(1,4)<2;1,0>\n"
        "mov (4) %0(1,12)<1> %1(1,13)<2;1,0>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline ushort get_d(const ushort3 src) {
    ushort d;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,9)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,2)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,11)<0;1,0>\n" 
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#elif MIN_SG_SIZE == 16
inline ushort2 get_qs(const ushort3 src) {
    ushort2 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<2;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,10)<2;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,19)<2;1,0>\n"
        "mov (4) %0(0,12)<1> %1(0,28)<2;1,0>\n" 

        "mov (4) %0(0,16)<1> %1(0,2)<2;1,0>\n" 
        "mov (4) %0(0,20)<1> %1(0,11)<2;1,0>\n"
        "mov (4) %0(0,24)<1> %1(0,20)<2;1,0>\n"
        "mov (4) %0(0,28)<1> %1(0,29)<2;1,0>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline ushort get_d(const ushort3 src) {
    ushort d;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,9)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,18)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(0,27)<0;1,0>\n" 
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#else
#error "Unsupported"
#endif

inline int2 repack(const ushort3 src) {
    const ushort2 quants = get_qs(src);
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

inline float mmvq_dot_product(
        const global block_q4_0 *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint tid = get_sub_group_local_id();
    const uint ib_k = ib_a - (tid * K_PER_ITER) / QUANT_K_Q8_1;
    const global ushort *ptr = (const global ushort *)(data_a + ib_k); 

    const ushort3 src = load(ptr);
    const int2 data_a_qs = as_int2(repack(src));
    const float da = convert_float(as_half(get_d(src)));

    int q_sum = 0;

    q_sum = IMAD(data_a_qs.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(data_a_qs.y, cache_b_qs[1], q_sum);

    // 2 quants per call => divide sums by 8/2 = 4
    return mul_q8_1(q_sum, da, cache_b_ds, 4);
}

)";

const char g_codeMmvqImpl_Q4_1[] = R"(
inline uint2 load(const global uint *data) {
    const uint tid = get_sub_group_local_id();
    uint2 src;
    src[0] = data[tid];
    src[1] = (tid < 4) ? data[16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline uint get_qs(const uint2 src) {
    uint qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<1;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,6)<1;1,0>\n"
        "mov (4) %0(1,0)<1> %1(1,3)<1;1,0>\n"
        "mov (4) %0(1,4)<1> %1(1,8)<1;1,0>\n" 
        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint get_dm(const uint2 src) {
    uint dm;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,5)<0;1,0>\n"
        "mov (4) %0(1,0)<1> %1(1,2)<0;1,0>\n"
        "mov (4) %0(1,4)<1> %1(1,7)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#elif MIN_SG_SIZE == 16
inline uint get_qs(const uint2 src) {
    uint qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<1;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,6)<1;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,11)<1;1,0>\n"
        "mov (4) %0(0,12)<1> %1(0,16)<1;1,0>\n" 
        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint get_dm(const uint2 src) {
    uint dm;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,5)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,10)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(0,15)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#else
#error "Unsupported"
#endif

inline int2 repack(const uint2 src) {
    const uint vui = get_qs(src);
    return (int2)(vui & 0x0F0F0F0F, (vui >> 4) & 0x0F0F0F0F);
}

inline float mul_q8_1(
        const int q_sum, 
        const float2 dma, 
        const float2 dsb, 
        const int sum_divisor) {
    return (float)q_sum * dma.x * dsb.x + dma.y * dsb.y / sum_divisor;
} 

inline float mmvq_dot_product(
        const global block_q4_1 *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint tid = get_sub_group_local_id();
    const uint ib_k = ib_a - (tid * K_PER_ITER) / QUANT_K_Q8_1;
    const global uint *ptr = (const global uint *)(data_a + ib_k); 

    const uint2 src = load(ptr);
    const int2 data_a_qs = as_int2(repack(src));
    const float2 dma = convert_float2(as_half2(get_dm(src)));

    int q_sum = 0;

    q_sum = IMAD(data_a_qs.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(data_a_qs.y, cache_b_qs[1], q_sum);

    // 2 quants per call => divide sums by 8/2 = 4
    return mul_q8_1(q_sum, dma, cache_b_ds, 4);
}

)";

const char g_codeMmvqImpl_Q5_0[] = R"(
inline ushort3 load(const global ushort *data) {
    const uint tid = get_sub_group_local_id();
    ushort3 src;
    unroll_for (int i = 0; i < 2; i++) {
        src[i] = data[i * 16 + tid];
    }
    src[2] = (tid < 12) ? data[2 * 16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline ushort2 get_qs(const ushort3 src) {
    ushort2 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,3)<2;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,14)<2;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,9)<2;1,0>\n"
        "mov (4) %0(0,12)<1> %1(2,4)<2;1,0>\n" 

        "mov (4) %0(1,0)<1> %1(0,4)<2;1,0>\n" 
        "mov (4) %0(1,4)<1> %1(0,15)<2;1,0>\n"
        "mov (4) %0(1,8)<1> %1(1,10)<2;1,0>\n"
        "mov (4) %0(1,12)<1> %1(2,5)<2;1,0>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline ushort2 get_qh(const ushort3 src) {
    ushort2 qh;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,12)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,7)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(2,2)<0;1,0>\n" 

        "mov (4) %0(1,0)<1> %1(0,2)<0;1,0>\n" 
        "mov (4) %0(1,4)<1> %1(0,13)<0;1,0>\n"
        "mov (4) %0(1,8)<1> %1(1,8)<0;1,0>\n"
        "mov (4) %0(1,12)<1> %1(2,3)<0;1,0>\n" 

        : "=rw"(qh)
        : "rw"(src));
    return qh;
}

inline ushort get_d(const ushort3 src) {
    ushort d;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,11)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,6)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(2,1)<0;1,0>\n" 
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#elif MIN_SG_SIZE == 16
inline ushort2 get_qs(const ushort3 src) {
    ushort2 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,3)<2;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,14)<2;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,25)<2;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,4)<2;1,0>\n" 

        "mov (4) %0(0,16)<1> %1(0,4)<2;1,0>\n" 
        "mov (4) %0(0,20)<1> %1(0,15)<2;1,0>\n"
        "mov (4) %0(0,24)<1> %1(0,26)<2;1,0>\n"
        "mov (4) %0(0,28)<1> %1(1,5)<2;1,0>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline ushort2 get_qh(const ushort3 src) {
    ushort2 qh;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,12)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,23)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,2)<0;1,0>\n" 

        "mov (4) %0(0,16)<1> %1(0,2)<0;1,0>\n" 
        "mov (4) %0(0,20)<1> %1(0,13)<0;1,0>\n"
        "mov (4) %0(0,24)<1> %1(0,24)<0;1,0>\n"
        "mov (4) %0(0,28)<1> %1(1,3)<0;1,0>\n" 

        : "=rw"(qh)
        : "rw"(src));
    return qh;
}

inline ushort get_d(const ushort3 src) {
    ushort d;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,11)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,22)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,1)<0;1,0>\n" 
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#else
#error "Unsupported"
#endif

inline int2 repack(const ushort3 src, const uint iqs) {
    const ushort2 vqs = get_qs(src);
    const ushort2 vqh = get_qh(src);
    const uint vui = as_uint(vqs);
    const int qh = (int)(((uint)vqh[1] << 16 | vqh[0]) >> (4 * iqs));
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

inline float mmvq_dot_product(
        const global block_q5_0 *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint tid = get_sub_group_local_id();
    const uint ib_k = ib_a - (tid * K_PER_ITER) / QUANT_K_Q8_1;
    const global ushort *ptr = (const global ushort *)(data_a + ib_k); 

    const ushort3 src = load(ptr);
    const int2 data_a_qs = as_int2(repack(src, iqs));
    const float da = convert_float(as_half(get_d(src)));

    int q_sum = 0;

    q_sum = IMAD(data_a_qs.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(data_a_qs.y, cache_b_qs[1], q_sum);

    // 2 quants per call => divide sums by 8/2 = 4
    return mul_q8_1(q_sum, da, cache_b_ds, 4);
}

)";

const char g_codeMmvqImpl_Q5_1[] = R"(
inline uint2 load(const global uint *data) {
    const uint tid = get_sub_group_local_id();
    uint2 src;
    src[0] = data[tid];
    src[1] = (tid < 8) ? data[16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline uint get_qs(const uint2 src) {
    uint qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,2)<1;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(1,0)<1;1,0>\n"
        "mov (4) %0(1,0)<1> %1(1,6)<1;1,0>\n"
        "mov (4) %0(1,4)<1> %1(2,4)<1;1,0>\n" 
        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint get_qh(const uint2 src) {
    uint qh;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,7)<0;1,0>\n"
        "mov (4) %0(1,0)<1> %1(1,5)<0;1,0>\n"
        "mov (4) %0(1,4)<1> %1(2,3)<0;1,0>\n" 
        : "=rw"(qh)
        : "rw"(src));
    return qh;
}

inline uint get_dm(const uint2 src) {
    uint dm;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,6)<0;1,0>\n"
        "mov (4) %0(1,0)<1> %1(1,4)<0;1,0>\n"
        "mov (4) %0(1,4)<1> %1(2,2)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#elif MIN_SG_SIZE == 16
inline uint get_qs(const uint2 src) {
    uint qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,2)<1;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,8)<1;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,14)<1;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,4)<1;1,0>\n" 
        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint get_qh(const uint2 src) {
    uint qh;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,7)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,13)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,3)<0;1,0>\n" 
        : "=rw"(qh)
        : "rw"(src));
    return qh;
}

inline uint get_dm(const uint2 src) {
    uint dm;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,6)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,12)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,2)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#else
#error "Unsupported"
#endif

inline int2 repack(const uint2 src, const uint iqs) {
    const uint vui = get_qs(src);
    const uint vqh = get_qh(src);
    const int qh = (int)(vqh >> (4 * iqs));
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

inline float mmvq_dot_product(
        const global block_q5_1 *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint tid = get_sub_group_local_id();
    const uint ib_k = ib_a - (tid * K_PER_ITER) / QUANT_K_Q8_1;
    const global uint *ptr = (const global uint *)(data_a + ib_k); 

    const uint2 src = load(ptr);
    const int2 data_a_qs = repack(src, iqs);
    const float2 dma = convert_float2(as_half2(get_dm(src)));

    int q_sum = 0;

    q_sum = IMAD(data_a_qs.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(data_a_qs.y, cache_b_qs[1], q_sum);

    // 2 quants per call => divide sums by 8/2 = 4
    return mul_q8_1(q_sum, dma, cache_b_ds, 4);
}

)";

const char g_codeMmvqImpl_Q8_0[] = R"(
inline ushort8 load(const global ushort *data) {
    const uint tid = get_sub_group_local_id();
    ushort8 src;
    unroll_for (int i = 0; i < 4; i++) {
        src[i] = data[i * 16 + tid];
    }
    src[4] = (tid < 4) ? data[4 * 16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline ushort4 get_qs(const ushort8 src) {
    ushort4 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<4;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(1,2)<4;1,0>\n"
        "mov (4) %0(0,8)<1> %1(2,3)<4;1,0>\n"
        "mov (4) %0(0,12)<1> %1(3,4)<4;1,0>\n" 

        "mov (4) %0(1,0)<1> %1(0,2)<4;1,0>\n" 
        "mov (4) %0(1,4)<1> %1(1,3)<4;1,0>\n"
        "mov (4) %0(1,8)<1> %1(2,4)<4;1,0>\n"
        "mov (4) %0(1,12)<1> %1(3,5)<4;1,0>\n" 

        "mov (4) %0(2,0)<1> %1(0,3)<4;1,0>\n" 
        "mov (4) %0(2,4)<1> %1(1,4)<4;1,0>\n"
        "mov (4) %0(2,8)<1> %1(2,5)<4;1,0>\n"
        "mov (4) %0(2,12)<1> %1(3,6)<4;1,0>\n" 

        "mov (4) %0(3,0)<1> %1(0,4)<4;1,0>\n" 
        "mov (4) %0(3,4)<1> %1(1,5)<4;1,0>\n"
        "mov (4) %0(3,8)<1> %1(2,6)<4;1,0>\n"
        "mov (4) %0(3,12)<1> %1(3,7)<4;1,0>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline ushort get_d(const ushort8 src) {
    ushort d;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,17)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,34)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(0,51)<0;1,0>\n" 
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#elif MIN_SG_SIZE == 16
inline ushort4 get_qs(const ushort8 src) {
    ushort4 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<4;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,18)<4;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,3)<4;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,20)<4;1,0>\n" 

        "mov (4) %0(0,16)<1> %1(0,2)<4;1,0>\n" 
        "mov (4) %0(0,20)<1> %1(0,19)<4;1,0>\n"
        "mov (4) %0(0,24)<1> %1(1,4)<4;1,0>\n"
        "mov (4) %0(0,28)<1> %1(1,21)<4;1,0>\n" 

        "mov (4) %0(1,0)<1> %1(0,3)<4;1,0>\n" 
        "mov (4) %0(1,4)<1> %1(0,20)<4;1,0>\n"
        "mov (4) %0(1,8)<1> %1(1,5)<4;1,0>\n"
        "mov (4) %0(1,12)<1> %1(1,22)<4;1,0>\n" 

        "mov (4) %0(1,16)<1> %1(0,4)<4;1,0>\n" 
        "mov (4) %0(1,20)<1> %1(0,21)<4;1,0>\n"
        "mov (4) %0(1,24)<1> %1(1,6)<4;1,0>\n"
        "mov (4) %0(1,28)<1> %1(1,23)<4;1,0>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline ushort get_d(const ushort8 src) {
    ushort d;
    __asm__(
        "mov (4) %0(0, 0)<1> %1(0, 0)<0;1,0>\n" 
        "mov (4) %0(0, 4)<1> %1(0, 17)<0;1,0>\n"
        "mov (4) %0(0, 8)<1> %1(0, 34)<0;1,0>\n"
        "mov (4) %0(0, 12)<1> %1(0, 51)<0;1,0>\n" 
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#else
#error "Unsupported"
#endif

inline int2 repack(const ushort8 src) {
    return as_int2(get_qs(src));
}

inline float mul_q8_1(
        const int q_sum, 
        const float da, 
        const float2 dsb) {
    return (float)q_sum * da * dsb.x;
} 

inline float mmvq_dot_product(
        const global block_q8_0 *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint tid = get_sub_group_local_id();
    const uint ib_k = ib_a - (tid * K_PER_ITER) / QUANT_K_Q8_1;
    const global ushort *ptr = (const global ushort *)(data_a + ib_k); 

    const ushort8 src = load(ptr);
    const int2 data_a_qs = repack(src);
    const float da = convert_float(as_half(get_d(src)));

    int q_sum = 0;

    q_sum = IMAD(data_a_qs[0], cache_b_qs[0], q_sum);
    q_sum = IMAD(data_a_qs[1], cache_b_qs[1], q_sum);

    return mul_q8_1(q_sum, da, cache_b_ds);
}

)";

const char g_codeMmvqImpl_Q2_K[] = R"(
inline uint2 load(const global uint *data) {
    const uint tid = get_sub_group_local_id();
    uint2 src;
    src[0] = data[tid];
    src[1] = (tid < 5) ? data[16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline uint4 get_qs(const uint2 src) {
    uint4 qs;
    __asm__(
        "mov (8) %0(0,0)<1> %1(0,4)<0;2,4>\n" 
        "mov (8) %0(1,0)<1> %1(1,4)<0;2,4>\n" 

        "mov (8) %0(2,0)<1> %1(0,5)<0;2,4>\n" 
        "mov (8) %0(3,0)<1> %1(1,5)<0;2,4>\n" 

        "mov (8) %0(4,0)<1> %1(0,6)<0;2,4>\n" 
        "mov (8) %0(5,0)<1> %1(1,6)<0;2,4>\n" 

        "mov (8) %0(6,0)<1> %1(0,7)<0;2,4>\n" 
        "mov (8) %0(7,0)<1> %1(1,7)<0;2,4>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint get_scales(const uint2 src) {
    uint scales;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,0)<1;4,0>\n" 
        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline uint get_dm(const uint2 src) {
    uint dm;
    __asm__(
        "mov (16) %0(0,0)<1> %1(2,4)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#elif MIN_SG_SIZE == 16
inline uint4 get_qs(const uint2 src) {
    uint4 qs;
    __asm__(
        "mov (8) %0(0,0)<1> %1(0,4)<0;2,4>\n" 
        "mov (8) %0(0,8)<1> %1(0,12)<0;2,4>\n" 

        "mov (8) %0(1,0)<1> %1(0,5)<0;2,4>\n" 
        "mov (8) %0(1,8)<1> %1(0,13)<0;2,4>\n" 

        "mov (8) %0(2,0)<1> %1(0,6)<0;2,4>\n" 
        "mov (8) %0(2,8)<1> %1(0,14)<0;2,4>\n" 

        "mov (8) %0(3,0)<1> %1(0,7)<0;2,4>\n" 
        "mov (8) %0(3,8)<1> %1(0,15)<0;2,4>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint get_scales(const uint2 src) {
    uint scales;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,0)<1;4,0>\n" 
        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline uint get_dm(const uint2 src) {
    uint dm;
    __asm__(
        "mov (16) %0(0,0)<1> %1(1,4)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#else
#error "Unsupported"
#endif

inline int4 repack4(
        const uint2 src, 
        const uint ib, 
        const uint iqs) {
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint qs_shift = ((iqs_k % 32) / 8) * 2;

    const uint4 qs = get_qs(src);

    return (int4)(
        (qs[0] >> qs_shift) & 0x03030303,
        (qs[1] >> qs_shift) & 0x03030303,
        (qs[2] >> qs_shift) & 0x03030303,
        (qs[3] >> qs_shift) & 0x03030303);
}

inline uchar get_scale(
        const uint2 src, 
        const uint ib, 
        const uint iqs) {
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint sc_shift = ((iqs_k / 4) % 4) * 8;

    const uint scales = get_scales(src);

    return (uchar)((scales >> sc_shift) & 0xFF);
}

float mmvq_dot_product(
        const global block_q2_K *data_a, 
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint ib_k = ib_a / 8;
    const global uint *ptr = (const global uint *)(data_a + ib_k); 

    const uint2 src = load(ptr);
    const int4 qs_a = repack4(src, ib_a, iqs * 4);
    const uchar scale = get_scale(src, ib_a, iqs * 4);
    const float2 dm = convert_float2(as_half2(get_dm(src)));
    const int scale_m = (int)(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.

    int sum_d = 0;
    int sum_m = 0;

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
inline ushort4 load(const global ushort *data) {
    const uint tid = get_sub_group_local_id();
    ushort4 src;
    unroll_for (int i = 0; i < 3; i++) {
        src[i] = data[i * 16 + tid];
    }
    src[3] = (tid < 7) ? data[3 * 16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline ushort8 get_qs(const ushort4 src) {
    ushort8 qs;
    __asm__(
        "mov (8) %0(0,0)<2> %1(1,0)<16;4,0>\n"
        "mov (8) %0(0,1)<2> %1(1,8)<16;4,0>\n"

        "mov (8) %0(1,0)<2> %1(1,1)<16;4,0>\n"
        "mov (8) %0(1,1)<2> %1(1,9)<16;4,0>\n"

        "mov (8) %0(2,0)<2> %1(1,2)<16;4,0>\n"
        "mov (8) %0(2,1)<2> %1(1,10)<16;4,0>\n"

        "mov (8) %0(3,0)<2> %1(1,3)<16;4,0>\n"
        "mov (8) %0(3,1)<2> %1(1,11)<16;4,0>\n"

        "mov (8) %0(4,0)<2> %1(1,4)<16;4,0>\n"
        "mov (8) %0(4,1)<2> %1(1,12)<16;4,0>\n"

        "mov (8) %0(5,0)<2> %1(1,5)<16;4,0>\n"
        "mov (8) %0(5,1)<2> %1(1,13)<16;4,0>\n"

        "mov (8) %0(6,0)<2> %1(1,6)<16;4,0>\n"
        "mov (8) %0(6,1)<2> %1(1,14)<16;4,0>\n"

        "mov (8) %0(7,0)<2> %1(1,7)<16;4,0>\n"
        "mov (8) %0(7,1)<2> %1(1,15)<16;4,0>\n"

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline ushort8 get_hmask(const ushort4 src) {
    ushort8 hmask;
    __asm__(
        "mov(8) %0(0,0)<2> %1(0,0)<0;1,0>\n"
        "mov(8) %0(0,1)<2> %1(0,8)<0;1,0>\n"

        "mov(8) %0(1,0)<2> %1(0,1)<0;1,0>\n"
        "mov(8) %0(1,1)<2> %1(0,9)<0;1,0>\n"

        "mov(8) %0(2,0)<2> %1(0,2)<0;1,0>\n"
        "mov(8) %0(2,1)<2> %1(0,10)<0;1,0>\n"

        "mov(8) %0(3,0)<2> %1(0,3)<0;1,0>\n"
        "mov(8) %0(3,1)<2> %1(0,11)<0;1,0>\n"

        "mov(8) %0(4,0)<2> %1(0,4)<0;1,0>\n"
        "mov(8) %0(4,1)<2> %1(0,12)<0;1,0>\n"

        "mov(8) %0(5,0)<2> %1(0,5)<0;1,0>\n"
        "mov(8) %0(5,1)<2> %1(0,13)<0;1,0>\n"

        "mov(8) %0(6,0)<2> %1(0,6)<0;1,0>\n"
        "mov(8) %0(6,1)<2> %1(0,14)<0;1,0>\n"

        "mov(8) %0(7,0)<2> %1(0,7)<0;1,0>\n"
        "mov(8) %0(7,1)<2> %1(0,15)<0;1,0>\n"

        : "=rw"(hmask)
        : "rw"(src));
    return hmask;
}

inline ushort2 get_scales(const ushort4 src) {
    // Note different patterns used for each output row
    ushort2 scales;
    __asm__(
        "mov (8) %0(0,0)<1> %1(3,0)<1;2,0>\n"
        "mov (8) %0(0,8)<1> %1(3,0)<1;2,0>\n"

        "mov (8) %0(1,0)<2> %1(3,4)<0;2,1>\n"
        "mov (8) %0(1,1)<2> %1(3,4)<0;2,1>\n"

        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline ushort get_d(const ushort4 src) {
    ushort d;
    __asm__(
        "mov (16) %0(0,0)<1> %1(3,6)<0;1,0>\n"
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#elif MIN_SG_SIZE == 16
inline ushort8 get_qs(const ushort4 src) {
    ushort8 qs;
    __asm__(
        "mov (8) %0(0,0)<2> %1(0,16)<16;4,0>\n"
        "mov (8) %0(0,1)<2> %1(0,24)<16;4,0>\n"

        "mov (8) %0(0,16)<2> %1(0,17)<16;4,0>\n"
        "mov (8) %0(0,17)<2> %1(0,25)<16;4,0>\n"

        "mov (8) %0(1,0)<2> %1(0,18)<16;4,0>\n"
        "mov (8) %0(1,1)<2> %1(0,26)<16;4,0>\n"

        "mov (8) %0(1,16)<2> %1(0,19)<16;4,0>\n"
        "mov (8) %0(1,17)<2> %1(0,27)<16;4,0>\n"

        "mov (8) %0(2,0)<2> %1(0,20)<16;4,0>\n"
        "mov (8) %0(2,1)<2> %1(0,28)<16;4,0>\n"

        "mov (8) %0(2,16)<2> %1(0,21)<16;4,0>\n"
        "mov (8) %0(2,17)<2> %1(0,29)<16;4,0>\n"

        "mov (8) %0(3,0)<2> %1(0,22)<16;4,0>\n"
        "mov (8) %0(3,1)<2> %1(0,30)<16;4,0>\n"

        "mov (8) %0(3,16)<2> %1(0,23)<16;4,0>\n"
        "mov (8) %0(3,17)<2> %1(0,31)<16;4,0>\n"

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline ushort8 get_hmask(const ushort4 src) {
    ushort8 hmask;
    __asm__(
        "mov(8) %0(0,0)<2> %1(0,0)<0;1,0>\n"
        "mov(8) %0(0,1)<2> %1(0,8)<0;1,0>\n"

        "mov(8) %0(0,16)<2> %1(0,1)<0;1,0>\n"
        "mov(8) %0(0,17)<2> %1(0,9)<0;1,0>\n"

        "mov(8) %0(1,0)<2> %1(0,2)<0;1,0>\n"
        "mov(8) %0(1,1)<2> %1(0,10)<0;1,0>\n"

        "mov(8) %0(1,16)<2> %1(0,3)<0;1,0>\n"
        "mov(8) %0(1,17)<2> %1(0,11)<0;1,0>\n"

        "mov(8) %0(2,0)<2> %1(0,4)<0;1,0>\n"
        "mov(8) %0(2,1)<2> %1(0,12)<0;1,0>\n"

        "mov(8) %0(2,16)<2> %1(0,5)<0;1,0>\n"
        "mov(8) %0(2,17)<2> %1(0,13)<0;1,0>\n"

        "mov(8) %0(3,0)<2> %1(0,6)<0;1,0>\n"
        "mov(8) %0(3,1)<2> %1(0,14)<0;1,0>\n"

        "mov(8) %0(3,16)<2> %1(0,7)<0;1,0>\n"
        "mov(8) %0(3,17)<2> %1(0,15)<0;1,0>\n"

        : "=rw"(hmask)
        : "rw"(src));
    return hmask;
}

inline ushort2 get_scales(const ushort4 src) {
    // Note different patterns used for each output row
    ushort2 scales;
    __asm__(
        "mov (8) %0(0,0)<1> %1(1,16)<1;2,0>\n"
        "mov (8) %0(0,8)<1> %1(1,16)<1;2,0>\n"

        "mov (8) %0(0,16)<2> %1(1,20)<0;2,1>\n"
        "mov (8) %0(0,17)<2> %1(1,20)<0;2,1>\n"

        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline ushort get_d(const ushort4 src) {
    ushort d;
    __asm__(
        "mov (16) %0(0,0)<1> %1(1,22)<0;1,0>\n"
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#else
#error "Unsupported"
#endif

inline int4 repack4(
        const ushort4 src,
        const uint ib, 
        const uint iqs) {
    const uint iqs_k = (ib % 8) * 8 + iqs;

    const uint qs_shift = ((iqs_k % 32) / 8) * 2;
    const uint hm_shift = iqs_k / 8;

    const ushort8 vqs = get_qs(src);
    const ushort8 vhmask = get_hmask(src);

    const uint4 qs = 
        (uint4)( 
            (uint)vqs[0] | ((uint)vqs[1] << 16),
            (uint)vqs[2] | ((uint)vqs[3] << 16),
            (uint)vqs[4] | ((uint)vqs[5] << 16),
            (uint)vqs[6] | ((uint)vqs[7] << 16));

    const uint4 hmask = 
        (uint4)( 
            (uint)vhmask[0] | ((uint)vhmask[1] << 16),
            (uint)vhmask[2] | ((uint)vhmask[3] << 16),
            (uint)vhmask[4] | ((uint)vhmask[5] << 16),
            (uint)vhmask[6] | ((uint)vhmask[7] << 16));

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

inline float get_d_scale(
        const ushort4 src,
        const uint ib, 
        const uint iqs) {
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 4;

    const ushort2 scs = get_scales(src);
    const half d = as_half(get_d(src));

    const uint sc_shift = (is % 2) * 8;
    const ushort sc0 = (scs[0] >> sc_shift) & 0xFF;    
    const ushort sc1 = (scs[1] >> sc_shift) & 0xFF;    

    const char scale = 
        (char)(((sc0 >> (4 * (is / 8))) & 0x0F0F) |
            (((sc1 >> (2 * (is / 4))) & 0x0303) << 4));

    return (float)d * (float)(scale - 32);
}

inline float mmvq_dot_product(
        const global block_q3_K *data_a, 
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint ib_k = ib_a / 8;
    const global ushort *ptr = (const global ushort *)(data_a + ib_k); 

    const ushort4 src = load(ptr);
    const int4 qs_a = repack4(src, ib_a, iqs * 4);
    const float d_scale = get_d_scale(src, ib_a, iqs * 4);

    int q_sum = 0;

    q_sum = IMAD(qs_a.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(qs_a.y, cache_b_qs[1], q_sum);
    q_sum = IMAD(qs_a.z, cache_b_qs[2], q_sum);
    q_sum = IMAD(qs_a.w, cache_b_qs[3], q_sum);

    return cache_b_ds.x * d_scale * (float)q_sum;
} 

)";

const char g_codeMmvqImpl_Q4_K[] = R"(
inline uint3 load(const global uint *data) {
    const uint tid = get_sub_group_local_id();
    uint3 src;
    src[0] = data[tid];
    src[1] = data[16 + tid];
    src[2] = (tid < 4) ? data[2 * 16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline uint4 get_qs(const uint3 src) {
    uint4 qs;
    __asm__(
        "mov (8) %0(0,0)<2> %1(0,4)<8;2,0>\n"
        "mov (8) %0(0,1)<2> %1(1,0)<8;2,0>\n"

        "mov (8) %0(2,0)<2> %1(0,5)<8;2,0>\n"
        "mov (8) %0(2,1)<2> %1(1,1)<8;2,0>\n"

        "mov (8) %0(4,0)<2> %1(0,6)<8;2,0>\n"
        "mov (8) %0(4,1)<2> %1(1,2)<8;2,0>\n"

        "mov (8) %0(6,0)<2> %1(0,7)<8;2,0>\n"
        "mov (8) %0(6,1)<2> %1(1,3)<8;2,0>\n"

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint3 get_scales(const uint3 src) {
    uint3 scales;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,1)<0;1,0>\n" 
        "mov (16) %0(2,0)<1> %1(0,2)<0;1,0>\n" 
        "mov (16) %0(4,0)<1> %1(0,3)<0;1,0>\n" 
        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline uint get_dm(const uint3 src) {
    uint dm;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#elif MIN_SG_SIZE == 16
inline uint4 get_qs(const uint3 src) {
    uint4 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,4)<0;2,4>\n"
        "mov (4) %0(0,4)<1> %1(0,12)<0;2,4>\n"
        "mov (4) %0(0,8)<1> %1(0,20)<0;2,4>\n"
        "mov (4) %0(0,12)<1> %1(0,28)<0;2,4>\n"

        "mov (4) %0(1,0)<1> %1(0,5)<0;2,4>\n"
        "mov (4) %0(1,4)<1> %1(0,13)<0;2,4>\n"
        "mov (4) %0(1,8)<1> %1(0,21)<0;2,4>\n"
        "mov (4) %0(1,12)<1> %1(0,29)<0;2,4>\n"

        "mov (4) %0(2,0)<1> %1(0,6)<0;2,4>\n"
        "mov (4) %0(2,4)<1> %1(0,14)<0;2,4>\n"
        "mov (4) %0(2,8)<1> %1(0,22)<0;2,4>\n"
        "mov (4) %0(2,12)<1> %1(0,30)<0;2,4>\n"

        "mov (4) %0(3,0)<1> %1(0,7)<0;2,4>\n"
        "mov (4) %0(3,4)<1> %1(0,15)<0;2,4>\n"
        "mov (4) %0(3,8)<1> %1(0,23)<0;2,4>\n"
        "mov (4) %0(3,12)<1> %1(0,31)<0;2,4>\n"

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint3 get_scales(const uint3 src) {
    uint3 scales;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,1)<0;1,0>\n" 
        "mov (16) %0(1,0)<1> %1(0,2)<0;1,0>\n" 
        "mov (16) %0(2,0)<1> %1(0,3)<0;1,0>\n" 
        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline uint get_dm(const uint3 src) {
    uint dm;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#else
#error "Unsupported"
#endif

inline int4 repack4(
        const uint3 src, 
        const uint ib, 
        const uint iqs) {
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;

    const uint4 qs = get_qs(src);

    const uint vals0 = (qs[0] >> qs_shift) & 0x0F0F0F0F;
    const uint vals1 = (qs[1] >> qs_shift) & 0x0F0F0F0F;
    const uint vals2 = (qs[2] >> qs_shift) & 0x0F0F0F0F;
    const uint vals3 = (qs[3] >> qs_shift) & 0x0F0F0F0F;

    return (int4)(vals0, vals1, vals2, vals3);
}

)";

const char g_codeMmvqImpl_Q5_K[] = R"(
inline uint3 load(const global uint *data) {
    const uint tid = get_sub_group_local_id();
    uint3 src;
    src[0] = data[tid];
    src[1] = data[16 + tid];
    src[2] = (tid < 12) ? data[2 * 16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline uint4 get_qs(const uint3 src) {
    uint4 qs;
    __asm__(
        "mov (8) %0(0,0)<2> %1(1,4)<8;2,0>\n"
        "mov (8) %0(0,1)<2> %1(2,0)<8;2,0>\n"

        "mov (8) %0(2,0)<2> %1(1,5)<8;2,0>\n"
        "mov (8) %0(2,1)<2> %1(2,1)<8;2,0>\n"

        "mov (8) %0(4,0)<2> %1(1,6)<8;2,0>\n"
        "mov (8) %0(4,1)<2> %1(2,2)<8;2,0>\n"

        "mov (8) %0(6,0)<2> %1(1,7)<8;2,0>\n"
        "mov (8) %0(6,1)<2> %1(2,3)<8;2,0>\n"

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint4 get_qh(const uint3 src) {
    uint4 qh;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,4)<0;2,4>\n"
        "mov (16) %0(2,0)<1> %1(0,5)<0;2,4>\n"
        "mov (16) %0(4,0)<1> %1(0,6)<0;2,4>\n"
        "mov (16) %0(6,0)<1> %1(0,7)<0;2,4>\n"
        : "=rw"(qh)
        : "rw"(src));
    return qh;
}

inline uint3 get_scales(const uint3 src) {
    uint3 scales;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,1)<0;1,0>\n" 
        "mov (16) %0(2,0)<1> %1(0,2)<0;1,0>\n" 
        "mov (16) %0(4,0)<1> %1(0,3)<0;1,0>\n" 
        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline uint get_dm(const uint3 src) {
    uint dm;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#elif MIN_SG_SIZE == 16
inline uint4 get_qs(const uint3 src) {
    uint4 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,12)<0;2,4>\n" 
        "mov (4) %0(0,4)<1> %1(1,4)<0;2,4>\n" 
        "mov (4) %0(0,8)<1> %1(1,12)<0;2,4>\n" 
        "mov (4) %0(0,12)<1> %1(2,4)<0;2,4>\n" 

        "mov (4) %0(1,0)<1> %1(0,13)<0;2,4>\n" 
        "mov (4) %0(1,4)<1> %1(1,5)<0;2,4>\n" 
        "mov (4) %0(1,8)<1> %1(1,13)<0;2,4>\n" 
        "mov (4) %0(1,12)<1> %1(2,5)<0;2,4>\n" 

        "mov (4) %0(2,0)<1> %1(0,14)<0;2,4>\n" 
        "mov (4) %0(2,4)<1> %1(1,6)<0;2,4>\n" 
        "mov (4) %0(2,8)<1> %1(1,14)<0;2,4>\n" 
        "mov (4) %0(2,12)<1> %1(2,6)<0;2,4>\n" 

        "mov (4) %0(3,0)<1> %1(0,15)<0;2,4>\n" 
        "mov (4) %0(3,4)<1> %1(1,7)<0;2,4>\n" 
        "mov (4) %0(3,8)<1> %1(1,15)<0;2,4>\n" 
        "mov (4) %0(3,12)<1> %1(2,7)<0;2,4>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uint4 get_qh(const uint3 src) {
    uint4 qh;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,4)<0;2,4>\n"
        "mov (16) %0(1,0)<1> %1(0,5)<0;2,4>\n"
        "mov (16) %0(2,0)<1> %1(0,6)<0;2,4>\n"
        "mov (16) %0(3,0)<1> %1(0,7)<0;2,4>\n"
        : "=rw"(qh)
        : "rw"(src));
    return qh;
}

inline uint3 get_scales(const uint3 src) {
    uint3 scales;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,1)<0;1,0>\n" 
        "mov (16) %0(1,0)<1> %1(0,2)<0;1,0>\n" 
        "mov (16) %0(2,0)<1> %1(0,3)<0;1,0>\n" 
        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline uint get_dm(const uint3 src) {
    uint dm;
    __asm__(
        "mov (16) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        : "=rw"(dm)
        : "rw"(src));
    return dm;
}

#else
#error "Unsupported"
#endif

inline int4 repack4(
        const uint3 src, 
        const uint ib, 
        const uint iqs) {
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint qs_shift = ((iqs_k % 16) / 8) * 4;
    const uint qh_shift = iqs_k / 8;

    const uint4 qs = get_qs(src);
    const uint4 qh = get_qh(src);

    return (int4)(
        ((qs[0] >> qs_shift) & 0x0F0F0F0F) | (((qh[0] >> qh_shift) & 0x01010101) << 4),
        ((qs[1] >> qs_shift) & 0x0F0F0F0F) | (((qh[1] >> qh_shift) & 0x01010101) << 4),
        ((qs[2] >> qs_shift) & 0x0F0F0F0F) | (((qh[2] >> qh_shift) & 0x01010101) << 4),
        ((qs[3] >> qs_shift) & 0x0F0F0F0F) | (((qh[3] >> qh_shift) & 0x01010101) << 4));
}

)";

const char g_codeMmvqImpl_Q45_K[] = R"(
// Common code for Q4_k and Q5_K

float2 get_dm_scale(
        const uint3 src,
        uint ib, 
        uint iqs) {
    const uint ib_k = ib / 8;
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 8;

    const uint3 scales = get_scales(src);
    const half2 dm = as_half2(get_dm(src));

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

    return convert_float2(dm) * convert_float2(scale_dm);
}

float mmvq_dot_product(
        const global A_TYPE *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint ib_k = ib_a / 8;
    const global uint *ptr = (const global uint *)(data_a + ib_k); 

    int q_sum = 0;

    const uint3 src = load(ptr);
    const int4 qs_a = repack4(src, ib_a, iqs * 4);
    const float2 dm_scale = get_dm_scale(src, ib_a, iqs * 4);

    q_sum = IMAD(qs_a.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(qs_a.y, cache_b_qs[1], q_sum);
    q_sum = IMAD(qs_a.z, cache_b_qs[2], q_sum);
    q_sum = IMAD(qs_a.w, cache_b_qs[3], q_sum);

    return cache_b_ds.x * dm_scale.x * (float)q_sum - dm_scale.y * (cache_b_ds.y / 2);
} 

)";

const char g_codeMmvqImpl_Q6_K[] = R"(
inline ushort8 load(const global ushort *data) {
    const uint tid = get_sub_group_local_id();
    ushort8 src;
    unroll_for (int i = 0; i < 6; i++) {
        src[i] = data[i * 16 + tid];
    }
    src[6] = (tid < 9) ? data[6 * 16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline ushort8 get_ql(const ushort8 src) {
    ushort8 ql;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<8;1,0>\n"
        "mov (4) %0(0,4)<1> %1(0,0)<8;1,0>\n"
        "mov (4) %0(0,8)<1> %1(2,0)<8;1,0>\n" 
        "mov (4) %0(0,12)<1> %1(2,0)<8;1,0>\n" 

        "mov (4) %0(1,0)<1> %1(0,1)<8;1,0>\n"
        "mov (4) %0(1,4)<1> %1(0,1)<8;1,0>\n"
        "mov (4) %0(1,8)<1> %1(2,1)<8;1,0>\n" 
        "mov (4) %0(1,12)<1> %1(2,1)<8;1,0>\n" 

        "mov (4) %0(2,0)<1> %1(0,2)<8;1,0>\n"
        "mov (4) %0(2,4)<1> %1(0,2)<8;1,0>\n"
        "mov (4) %0(2,8)<1> %1(2,2)<8;1,0>\n" 
        "mov (4) %0(2,12)<1> %1(2,2)<8;1,0>\n" 

        "mov (4) %0(3,0)<1> %1(0,3)<8;1,0>\n"
        "mov (4) %0(3,4)<1> %1(0,3)<8;1,0>\n"
        "mov (4) %0(3,8)<1> %1(2,3)<8;1,0>\n" 
        "mov (4) %0(3,12)<1> %1(2,3)<8;1,0>\n" 

        "mov (4) %0(4,0)<1> %1(0,4)<8;1,0>\n"
        "mov (4) %0(4,4)<1> %1(0,4)<8;1,0>\n"
        "mov (4) %0(4,8)<1> %1(2,4)<8;1,0>\n" 
        "mov (4) %0(4,12)<1> %1(2,4)<8;1,0>\n" 

        "mov (4) %0(5,0)<1> %1(0,5)<8;1,0>\n"
        "mov (4) %0(5,4)<1> %1(0,5)<8;1,0>\n"
        "mov (4) %0(5,8)<1> %1(2,5)<8;1,0>\n" 
        "mov (4) %0(5,12)<1> %1(2,5)<8;1,0>\n" 

        "mov (4) %0(6,0)<1> %1(0,6)<8;1,0>\n"
        "mov (4) %0(6,4)<1> %1(0,6)<8;1,0>\n"
        "mov (4) %0(6,8)<1> %1(2,6)<8;1,0>\n" 
        "mov (4) %0(6,12)<1> %1(2,6)<8;1,0>\n" 

        "mov (4) %0(7,0)<1> %1(0,7)<8;1,0>\n"
        "mov (4) %0(7,4)<1> %1(0,7)<8;1,0>\n"
        "mov (4) %0(7,8)<1> %1(2,7)<8;1,0>\n" 
        "mov (4) %0(7,12)<1> %1(2,7)<8;1,0>\n" 

        : "=rw"(ql)
        : "rw"(src));
    return ql;
}

inline ushort8 get_qh(const ushort8 src) {
    ushort8 qh;
    __asm__(
        "mov (8) %0(0,0)<2> %1(4,0)<16;4,0>\n"
        "mov (8) %0(0,1)<2> %1(4,8)<16;4,0>\n"

        "mov (8) %0(1,0)<2> %1(4,1)<16;4,0>\n"
        "mov (8) %0(1,1)<2> %1(4,9)<16;4,0>\n"

        "mov (8) %0(2,0)<2> %1(4,2)<16;4,0>\n"
        "mov (8) %0(2,1)<2> %1(4,10)<16;4,0>\n"

        "mov (8) %0(3,0)<2> %1(4,3)<16;4,0>\n"
        "mov (8) %0(3,1)<2> %1(4,11)<16;4,0>\n"

        "mov (8) %0(4,0)<2> %1(4,4)<16;4,0>\n"
        "mov (8) %0(4,1)<2> %1(4,12)<16;4,0>\n"

        "mov (8) %0(5,0)<2> %1(4,5)<16;4,0>\n"
        "mov (8) %0(5,1)<2> %1(4,13)<16;4,0>\n"

        "mov (8) %0(6,0)<2> %1(4,6)<16;4,0>\n"
        "mov (8) %0(6,1)<2> %1(4,14)<16;4,0>\n"

        "mov (8) %0(7,0)<2> %1(4,7)<16;4,0>\n"
        "mov (8) %0(7,1)<2> %1(4,15)<16;4,0>\n"

        : "=rw"(qh)
        : "rw"(src));
    return qh;
}

inline ushort get_scales(const ushort8 src) {
    ushort scales;
    __asm__(
        "mov (16) %0(0,0)<1> %1(6,0)<1;2,0>\n"
        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline ushort get_d(const ushort8 src) {
    ushort d;
    __asm__(
        "mov (16) %0(0,0)<1> %1(6,8)<0;1,0>\n"
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#elif MIN_SG_SIZE == 16
inline ushort8 get_ql(const ushort8 src) {
    ushort8 ql;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<8;1,0>\n"
        "mov (4) %0(0,4)<1> %1(0,0)<8;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,0)<8;1,0>\n" 
        "mov (4) %0(0,12)<1> %1(1,0)<8;1,0>\n" 

        "mov (4) %0(0,16)<1> %1(0,1)<8;1,0>\n"
        "mov (4) %0(0,20)<1> %1(0,1)<8;1,0>\n"
        "mov (4) %0(0,24)<1> %1(1,1)<8;1,0>\n" 
        "mov (4) %0(0,28)<1> %1(1,1)<8;1,0>\n" 

        "mov (4) %0(1,0)<1> %1(0,2)<8;1,0>\n"
        "mov (4) %0(1,4)<1> %1(0,2)<8;1,0>\n"
        "mov (4) %0(1,8)<1> %1(1,2)<8;1,0>\n" 
        "mov (4) %0(1,12)<1> %1(1,2)<8;1,0>\n" 

        "mov (4) %0(1,16)<1> %1(0,3)<8;1,0>\n"
        "mov (4) %0(1,20)<1> %1(0,3)<8;1,0>\n"
        "mov (4) %0(1,24)<1> %1(1,3)<8;1,0>\n" 
        "mov (4) %0(1,28)<1> %1(1,3)<8;1,0>\n" 

        "mov (4) %0(2,0)<1> %1(0,4)<8;1,0>\n"
        "mov (4) %0(2,4)<1> %1(0,4)<8;1,0>\n"
        "mov (4) %0(2,8)<1> %1(1,4)<8;1,0>\n" 
        "mov (4) %0(2,12)<1> %1(1,4)<8;1,0>\n" 

        "mov (4) %0(2,16)<1> %1(0,5)<8;1,0>\n"
        "mov (4) %0(2,20)<1> %1(0,5)<8;1,0>\n"
        "mov (4) %0(2,24)<1> %1(1,5)<8;1,0>\n" 
        "mov (4) %0(2,28)<1> %1(1,5)<8;1,0>\n" 

        "mov (4) %0(3,0)<1> %1(0,6)<8;1,0>\n"
        "mov (4) %0(3,4)<1> %1(0,6)<8;1,0>\n"
        "mov (4) %0(3,8)<1> %1(1,6)<8;1,0>\n" 
        "mov (4) %0(3,12)<1> %1(1,6)<8;1,0>\n" 

        "mov (4) %0(3,16)<1> %1(0,7)<8;1,0>\n"
        "mov (4) %0(3,20)<1> %1(0,7)<8;1,0>\n"
        "mov (4) %0(3,24)<1> %1(1,7)<8;1,0>\n" 
        "mov (4) %0(3,28)<1> %1(1,7)<8;1,0>\n" 

        : "=rw"(ql)
        : "rw"(src));
    return ql;
}

inline ushort8 get_qh(const ushort8 src) {
    ushort8 qh;
    __asm__(
        "mov (8) %0(0,0)<2> %1(2,0)<16;4,0>\n"
        "mov (8) %0(0,1)<2> %1(2,8)<16;4,0>\n"

        "mov (8) %0(0,16)<2> %1(2,1)<16;4,0>\n"
        "mov (8) %0(0,17)<2> %1(2,9)<16;4,0>\n"

        "mov (8) %0(1,0)<2> %1(2,2)<16;4,0>\n"
        "mov (8) %0(1,1)<2> %1(2,10)<16;4,0>\n"

        "mov (8) %0(1,16)<2> %1(2,3)<16;4,0>\n"
        "mov (8) %0(1,17)<2> %1(2,11)<16;4,0>\n"

        "mov (8) %0(2,0)<2> %1(2,4)<16;4,0>\n"
        "mov (8) %0(2,1)<2> %1(2,12)<16;4,0>\n"

        "mov (8) %0(2,16)<2> %1(2,5)<16;4,0>\n"
        "mov (8) %0(2,17)<2> %1(2,13)<16;4,0>\n"

        "mov (8) %0(3,0)<2> %1(2,6)<16;4,0>\n"
        "mov (8) %0(3,1)<2> %1(2,14)<16;4,0>\n"

        "mov (8) %0(3,16)<2> %1(2,7)<16;4,0>\n"
        "mov (8) %0(3,17)<2> %1(2,15)<16;4,0>\n"

        : "=rw"(qh)
        : "rw"(src));
    return qh;
}

inline ushort get_scales(const ushort8 src) {
    ushort scales;
    __asm__(
        "mov (16) %0(0,0)<1> %1(3,0)<1;2,0>\n"
        : "=rw"(scales)
        : "rw"(src));
    return scales;
}

inline ushort get_d(const ushort8 src) {
    ushort d;
    __asm__(
        "mov (16) %0(0,0)<1> %1(3,8)<0;1,0>\n"
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#else
#error "Unsupported"
#endif

inline int4 repack4(
        const ushort8 src, 
        const uint ib, 
        const uint iqs) {
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint ql_shift = ((iqs_k % 32) / 16) * 4;
    const uint qh_shift = ((iqs_k % 32) / 8) * 2;

    const ushort8 vql = get_ql(src);
    const ushort8 vqh = get_qh(src);

    const uint4 ql = 
        (uint4)( 
            (uint)vql[0] | ((uint)vql[1] << 16),
            (uint)vql[2] | ((uint)vql[3] << 16),
            (uint)vql[4] | ((uint)vql[5] << 16),
            (uint)vql[6] | ((uint)vql[7] << 16));

    const uint4 qh = 
        (uint4)( 
            (uint)vqh[0] | ((uint)vqh[1] << 16),
            (uint)vqh[2] | ((uint)vqh[3] << 16),
            (uint)vqh[4] | ((uint)vqh[5] << 16),
            (uint)vqh[6] | ((uint)vqh[7] << 16));

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

inline float get_d_scale(
        const ushort8 src,
        const uint ib, 
        const uint iqs) {
    const uint iqs_k = (ib % 8) * 8 + iqs;
    const uint is = iqs_k / 4;

    const ushort scales = get_scales(src);
    const uint sc_shift = (is % 2) * 8;

    const char sc = (char)((scales >> sc_shift) & 0xFF);
    const half d = as_half(get_d(src));
    return (float)d * (float)sc;
}

inline float mmvq_dot_product(
        const global block_q6_K *data_a, 
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint ib_k = ib_a / 8;
    const global ushort *ptr = (const global ushort *)(data_a + ib_k); 

    const ushort8 src = load(ptr);
    const int4 qs_a = repack4(src, ib_a, iqs * 4);
    const float d_scale = get_d_scale(src, ib_a, iqs * 4);

    int q_sum = 0;

    q_sum = IMAD(qs_a.x, cache_b_qs[0], q_sum);
    q_sum = IMAD(qs_a.y, cache_b_qs[1], q_sum);
    q_sum = IMAD(qs_a.z, cache_b_qs[2], q_sum);
    q_sum = IMAD(qs_a.w, cache_b_qs[3], q_sum);

    return cache_b_ds.x * d_scale * (float)q_sum;
} 

)";

const char g_codeMmvqImpl_Mxfp4[] = R"(
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

inline uchar8 load(const global uchar *data) {
    const uint tid = get_sub_group_local_id();
    uchar8 src;
    unroll_for (int i = 0; i < 4; i++) {
        src[i] = data[i * 16 + tid];
    }
    src[4] = (tid < 4) ? data[4 * 16 + tid] : 0;
    return src;
}

#if MIN_SG_SIZE == 8
inline uchar4 get_qs(const uchar8 src) {
    uchar4 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<4;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,18)<4;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,3)<4;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,20)<4;1,0>\n" 

        "mov (4) %0(0,16)<1> %1(0,2)<4;1,0>\n" 
        "mov (4) %0(0,20)<1> %1(0,19)<4;1,0>\n"
        "mov (4) %0(0,24)<1> %1(1,4)<4;1,0>\n"
        "mov (4) %0(0,28)<1> %1(1,21)<4;1,0>\n" 

        "mov (4) %0(1,0)<1> %1(0,3)<4;1,0>\n" 
        "mov (4) %0(1,4)<1> %1(0,20)<4;1,0>\n"
        "mov (4) %0(1,8)<1> %1(1,5)<4;1,0>\n"
        "mov (4) %0(1,12)<1> %1(1,22)<4;1,0>\n" 

        "mov (4) %0(1,16)<1> %1(0,4)<4;1,0>\n" 
        "mov (4) %0(1,20)<1> %1(0,21)<4;1,0>\n"
        "mov (4) %0(1,24)<1> %1(1,6)<4;1,0>\n"
        "mov (4) %0(1,28)<1> %1(1,23)<4;1,0>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uchar get_d(const uchar8 src) {
    uchar d;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,17)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(1,2)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(1,19)<0;1,0>\n" 
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#elif MIN_SG_SIZE == 16
inline uchar4 get_qs(const uchar8 src) {
    uchar4 qs;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,1)<4;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,18)<4;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,35)<4;1,0>\n"
        "mov (4) %0(0,12)<1> %1(0,52)<4;1,0>\n" 

        "mov (4) %0(0,16)<1> %1(0,2)<4;1,0>\n" 
        "mov (4) %0(0,20)<1> %1(0,19)<4;1,0>\n"
        "mov (4) %0(0,24)<1> %1(0,36)<4;1,0>\n"
        "mov (4) %0(0,28)<1> %1(0,53)<4;1,0>\n" 

        "mov (4) %0(0,32)<1> %1(0,3)<4;1,0>\n" 
        "mov (4) %0(0,36)<1> %1(0,20)<4;1,0>\n"
        "mov (4) %0(0,40)<1> %1(0,37)<4;1,0>\n"
        "mov (4) %0(0,44)<1> %1(0,54)<4;1,0>\n" 

        "mov (4) %0(0,48)<1> %1(0,4)<4;1,0>\n" 
        "mov (4) %0(0,52)<1> %1(0,21)<4;1,0>\n"
        "mov (4) %0(0,56)<1> %1(0,38)<4;1,0>\n"
        "mov (4) %0(0,60)<1> %1(0,55)<4;1,0>\n" 

        : "=rw"(qs)
        : "rw"(src));
    return qs;
}

inline uchar get_d(const uchar8 src) {
    uchar d;
    __asm__(
        "mov (4) %0(0,0)<1> %1(0,0)<0;1,0>\n" 
        "mov (4) %0(0,4)<1> %1(0,17)<0;1,0>\n"
        "mov (4) %0(0,8)<1> %1(0,34)<0;1,0>\n"
        "mov (4) %0(0,12)<1> %1(0,51)<0;1,0>\n" 
        : "=rw"(d)
        : "rw"(src));
    return d;
}

#else
#error "Unsupported"
#endif

inline int2 repack(const uchar8 src) {
    const uint qs = as_uint(get_qs(src));

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
        const float2 dsb) {
    return da * dsb.x * (float)q_sum * 0.5f;
} 

inline float mmvq_dot_product(
        const global block_mxfp4 *data_a,
        const int cache_b_qs[K_PER_ITER / 4],
        const float2 cache_b_ds, 
        const uint ib_a, 
        const uint iqs) {
    const uint tid = get_sub_group_local_id();
    const uint ib_k = ib_a - (tid * K_PER_ITER) / QUANT_K_Q8_1;
    const global uchar *ptr = (const global uchar *)(data_a + ib_k); 

    const uchar8 src = load(ptr);
    const int2 data_a_qs = repack(src);
    const float da = e8m0_to_fp32(get_d(src));

    int q_sum = 0;

    q_sum = IMAD(data_a_qs[0], cache_b_qs[0], q_sum);
    q_sum = IMAD(data_a_qs[1], cache_b_qs[1], q_sum);

    return mul_q8_1(q_sum, da, cache_b_ds);
}

)";

//
//    MulMatQuantVecV2Opt
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
            // TODO: Provide for tail load
            temp[j][n] += 
                mmvq_dot_product(
                    data_a,
                    cache_b_qs,
                    cache_b_ds,
                    a_block_idx, 
                    b_qs_idx);
            ibi += ncols;
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
#if 0 // TODO: Revise this
    if (num_iters * K_PER_ITER * BLOCK_SIZE + K_PER_ITER * tid < ncols) {
        num_iters++;
    }
#endif

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
        first_row, 
        num_rows);
} 

#else
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
        const uint ids_dim3,
        const uint b_dim2,
        const uint ids_i2,
        const uint ids_stride2,
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
#if 0 // TODO: Revise this
    if (num_iters * K_PER_ITER * BLOCK_SIZE + K_PER_ITER * tid < ncols) {
        num_iters++;
    }
#endif

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
        uint ncols,
        uint stride_a,
        uint stride_b,
        uint stride_d,
        uint batch_stride_a,
        uint batch_stride_b,
        uint batch_stride_d, 
        const uint ids_dim3,
        const uint b_dim2,
        const uint ids_stride2
        SHAPE_INFO_ARGS) {

    data_a += A_BASE / sizeof(A_TYPE);
    data_b += B_BASE / sizeof(B_TYPE);
    data_ids += IDS_BASE;
    data_d += D_BASE;

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
        first_row, 
        num_rows);
} 

#endif

)";

} // namespace

//
//    MulMatQuantVecV2OptImpl
//

const char *MulMatQuantVecV2OptImpl_Q4_0_Code() {
    return g_codeMmvqImpl_Q4_0;
}

const char *MulMatQuantVecV2OptImpl_Q4_1_Code() {
    return g_codeMmvqImpl_Q4_1;
}

const char *MulMatQuantVecV2OptImpl_Q5_0_Code() {
    return g_codeMmvqImpl_Q5_0;
}

const char *MulMatQuantVecV2OptImpl_Q5_1_Code() {
    return g_codeMmvqImpl_Q5_1;
}

const char *MulMatQuantVecV2OptImpl_Q8_0_Code() {
    return g_codeMmvqImpl_Q8_0;
}

const char *MulMatQuantVecV2OptImpl_Q2_K_Code() {
    return g_codeMmvqImpl_Q2_K;
}

const char *MulMatQuantVecV2OptImpl_Q3_K_Code() {
    return g_codeMmvqImpl_Q3_K;
}

const char *MulMatQuantVecV2OptImpl_Q4_K_Code() {
    return g_codeMmvqImpl_Q4_K;
}

const char *MulMatQuantVecV2OptImpl_Q5_K_Code() {
    return g_codeMmvqImpl_Q5_K;
}

const char *MulMatQuantVecV2OptImpl_Q45_K_Code() {
    return g_codeMmvqImpl_Q45_K;
}

const char *MulMatQuantVecV2OptImpl_Q6_K_Code() {
    return g_codeMmvqImpl_Q6_K;
}

const char *MulMatQuantVecV2OptImpl_Mxfp4_Code() {
    return g_codeMmvqImpl_Mxfp4;
}

//
//    MulMatQuantVecV2Opt
//

const char *MulMatQuantVecV2OptKernelCode() {
    return g_kernelCodeMmvq;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

