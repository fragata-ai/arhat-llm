/* 
* MIT License
*
* Copyright (c) 2026 FRAGATA COMPUTER SYSTEMS AG
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
#include <string>
#include <limits>

#include "dnnl.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

//
//    Arithmetics
//

template<typename T>
T DivUp(T a, T b) {
    return (a + b - 1) / b;
}

template<typename T>
T RndUp(T a, T b) {
    return DivUp(a, b) * b;
}

template<typename T>
T RndUpPow2(const T a) {
    if (a <= 0)
        return T(1);
    else {
        T b = a - 1;
        for (size_t v = 1; v < sizeof(T) * CHAR_BIT; v <<= 1) {
            b |= (b >> v);
        }
        return T(b + 1);
    }
}

template<typename T>
T MaxDiv(T a, T b) {
    T div = b;
    while (div > 1) {
        if (a % div == 0) {
            return div;
        }
        div--;
    }
    return div;
}

void MakeFastDiv(int64_t x, uint32_t &fd0, uint32_t &fd1);

//
//    Formatting
//

std::string FormatInt(int64_t value);
std::string FormatFloat(float value);
std::string FormatType(dnnl::memory::data_type dt);

} // namespace ocl
} // namespace onednn
} // namespace arhat

