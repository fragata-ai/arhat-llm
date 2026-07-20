/* 
* MIT License
*
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

#pragma once

#include <cstdint>

namespace arhat {
namespace onednn {
namespace base {

//
//    Common constants
//

constexpr int QK_K = 256;
constexpr int K_SCALE_SIZE = 12;

//
//    Q2_K
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight 

struct Block_Q2_K {
    uint8_t scales[QK_K / 16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K / 4];      // quants
    uint16_t d;                // super-block scale for quantized scales (f16)
    uint16_t dmin;             // super-block scale for quantized mins (f16)
};

static_assert(
    sizeof(Block_Q2_K) == 2 * sizeof(uint16_t) + QK_K / 16 + QK_K / 4, 
    "wrong Q2_K block size/padding"); 

//
//    Q3_K
//

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight 

struct Block_Q3_K {
    uint8_t hmask[QK_K / 8]; // quants - high bit
    uint8_t qs[QK_K / 4];    // quants - low 2 bits
    uint8_t scales[12];      // scales, quantized with 6 bits
    uint16_t d;              // super-block scale (f16)
};

static_assert(
    sizeof(Block_Q3_K) == sizeof(uint16_t) + QK_K / 4 + QK_K / 8 + 12, 
    "wrong Q3_K block size/padding"); 

//
//    Q4_0
//

constexpr int QK4_0 = 32;

struct Block_Q4_0 {
    uint16_t d;            // delta (f16)
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
};

static_assert(
    sizeof(Block_Q4_0) == sizeof(uint16_t) + QK4_0 / 2, 
    "wrong Q4_0 block size/padding"); 

//
//    QK4_1
//

constexpr int QK4_1 = 32;

struct Block_Q4_1 {
    uint16_t d;            // delta (f16)
    uint16_t m;            // min (f16)
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
};

static_assert(
    sizeof(Block_Q4_1) == 2 * sizeof(uint16_t) + QK4_1 / 2, 
    "wrong Q4_1 block size/padding"); 

//
//    Q4_K
//

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight

struct Block_Q4_K {
    uint16_t d;                   // super-block scale for quantized scales (f16)
    uint16_t dmin;                // super-block scale for quantized mins (f16)
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];         // 4-bit quants
};

static_assert(
    sizeof(Block_Q4_K) == 2 * sizeof(uint16_t) + K_SCALE_SIZE + QK_K / 2, 
    "wrong Q4_K block size/padding"); 

//
//    Q5_0
//

constexpr int QK5_0 = 32;

struct Block_Q5_0 {
    uint16_t d;            // delta (f16)
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
};

static_assert(
    sizeof(Block_Q5_0) == sizeof(uint16_t) + 4 + QK5_0 / 2, 
    "wrong Q5_0 block size/padding");
 
//
//    Q5_1
//

constexpr int QK5_1 = 32;

struct Block_Q5_1 {
    uint16_t d;            // delta (fp16)
    uint16_t m;            // min (fp16)
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
};

static_assert(
    sizeof(Block_Q5_1) == 2 * sizeof(uint16_t) + 4 + QK5_1 / 2, 
    "wrong Q5_1 block size/padding");
 
//
//    Q5_K
//

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight

struct Block_Q5_K {
    uint16_t d;                   // super-block scale for quantized scales (f16)
    uint16_t dmin;                // super-block scale for quantized mins (f16)
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
};

static_assert(
    sizeof(Block_Q5_K) == 2 * sizeof(uint16_t) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8, 
    "wrong q5_K block size/padding"); 

//
//    Q6_K
//

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight

struct Block_Q6_K {
    uint8_t ql[QK_K / 2];     // quants, lower 4 bits
    uint8_t qh[QK_K / 4];     // quants, upper 2 bits
    int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
    uint16_t d;               // super-block scale (f16)
};

static_assert(
    sizeof(Block_Q6_K) == QK_K / 2 + QK_K / 4 + QK_K / 16 + sizeof(uint16_t),
    "wrong Q6_K block size/padding");

//
//    Q8_0
//

constexpr int QK8_0 = 32;

struct Block_Q8_0 {
    uint16_t d;       // delta (f16)
    int8_t qs[QK8_0]; // quants
};

static_assert(
    sizeof(Block_Q8_0) == sizeof(uint16_t) + QK8_0, 
    "wrong Q8_0 block size/padding"); 

//
//    Q8_1
//

constexpr int QK8_1 = 32;

struct Block_Q8_1 {
    uint16_t d;       // delta (f16)
    uint16_t s;       // d * sum(qs[*]) (f16)
    int8_t qs[QK8_1]; // quants
};

static_assert(
    sizeof(Block_Q8_1) == 2 * sizeof(uint16_t) + QK8_1, 
    "wrong Q8_1 block size/padding");

//
//    MFXP4
//

constexpr int QK_MXFP4 = 32;

struct Block_Mxfp4 {
    uint8_t e; // E8M0
    uint8_t qs[QK_MXFP4 / 2];
};

static_assert(
    sizeof(Block_Mxfp4) == sizeof(uint8_t) + QK_MXFP4 / 2, 
    "wrong mxfp4 block size/padding");
 
} // namespace base
} // namespace onednn
} // namespace arhat

