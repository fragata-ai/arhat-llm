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

#include <cassert>
#include <string>

#include "arhat/onednn/base/runtime.hpp"

#include "arhat/onednn/gpu/quant_traits.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

QuantTraits GetQuantTraits(base::QuantMode quant) {
    using namespace QuantConst;
    switch (quant) {
    case base::QuantMode::Q4_0:
        return {QK4_0, QR4_0, QI4_0};
    case base::QuantMode::Q4_1:
        return {QK4_1, QR4_1, QI4_1};
    case base::QuantMode::Q5_0:
        return {QK5_0, QR5_0, QI5_0};
    case base::QuantMode::Q5_1:
        return {QK5_1, QR5_1, QI5_1};
    case base::QuantMode::Q8_0:
        return {QK8_0, QR8_0, QI8_0};
    case base::QuantMode::Q2_K:
        return {QK_K, QR2_K, QI2_K};
    case base::QuantMode::Q3_K:
        return {QK_K, QR3_K, QI3_K};
    case base::QuantMode::Q4_K:
        return {QK_K, QR4_K, QI4_K};
    case base::QuantMode::Q5_K:
        return {QK_K, QR5_K, QI5_K};
    case base::QuantMode::Q6_K:
        return {QK_K, QR6_K, QI6_K};
    case base::QuantMode::MXFP4:
        return {QK_MXFP4, QR_MXFP4, QI_MXFP4};
    default:
        assert(false);
        return {};
    }
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

