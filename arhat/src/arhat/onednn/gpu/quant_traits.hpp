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

#pragma once

#include <string>

#include "arhat/onednn/base/runtime.hpp"
#include "arhat/onednn/base/quant.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace QuantConst {
    constexpr int QK_K = base::QK_K;

    constexpr int QK4_0 = base::QK4_0;
    constexpr int QR4_0 = 2;
    constexpr int QI4_0 = QK4_0 / (4 * QR4_0);

    constexpr int QK4_1 = base::QK4_1;
    constexpr int QR4_1 = 2;
    constexpr int QI4_1 = QK4_1 / (4 * QR4_1);

    constexpr int QK5_0 = base::QK5_0;
    constexpr int QR5_0 = 2;
    constexpr int QI5_0 = QK5_0 / (4 * QR5_0);

    constexpr int QK5_1 = base::QK5_1;
    constexpr int QR5_1 = 2; 
    constexpr int QI5_1 = QK5_1 / (4 * QR5_1);

    constexpr int QK8_0 = base::QK8_0;
    constexpr int QR8_0 = 1;
    constexpr int QI8_0 = QK8_0 / (4 * QR8_0);

    constexpr int QK8_1 = base::QK8_1;
    constexpr int QR8_1 = 1;
    constexpr int QI8_1 = QK8_1 / (4 * QR8_1);

    constexpr int QR2_K = 4;
    constexpr int QI2_K = QK_K / (4 * QR2_K);

    constexpr int QR3_K = 4;
    constexpr int QI3_K = QK_K / (4 * QR3_K);

    constexpr int QR4_K = 2;
    constexpr int QI4_K = QK_K / (4 * QR4_K);

    constexpr int QR5_K = 2;
    constexpr int QI5_K = QK_K / (4 * QR5_K);

    constexpr int QR6_K = 2;
    constexpr int QI6_K = QK_K / (4 * QR6_K);

    constexpr int QK_MXFP4 = base::QK_MXFP4;
    constexpr int QR_MXFP4 = 2;
    constexpr int QI_MXFP4 = QK_MXFP4 / (4 * QR_MXFP4);
}

struct QuantTraits {
    int qk = 0;
    int qr = 0;
    int qi = 0;
};

QuantTraits GetQuantTraits(base::QuantMode quant);

} // namespace gpu
} // namespace onednn
} // namespace arhat

