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

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <string>
#include <limits>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/util.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

//
//    Arithmetics
//

void MakeFastDiv(int64_t x, uint32_t &fd0, uint32_t &fd1) {
    assert(x > 0 && x <= std::numeric_limits<uint32_t>::max());
    uint32_t x32 = uint32_t(x);
    fd1 = 0;
    while (fd1 < 32 && (uint32_t(1) << fd1) < x32) {
        fd1++;
    }
    fd0 = uint32_t((uint64_t(1) << 32) * ((uint64_t(1) << fd1) - x) / x + 1);
}

//
//    Formatting
//

std::string FormatInt(int64_t value) {
    std::string result;
    if (value == std::numeric_limits<int32_t>::min()) {
        result = "(-2147483647-1)";
    } else if (value == std::numeric_limits<int64_t>::min()) {
        result = "(-9223372036854775807L-1)";
    } else {
        result = std::to_string(value);
        if (value > INT_MAX || value < INT_MIN) {
            result += "L";
        }
    }
    return result;
}

std::string FormatFloat(float value) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = value;
    char buf[64];
    snprintf(buf, sizeof(buf), "as_float(0x%x)", (unsigned int)u.i);
    return buf;
}

std::string FormatType(dnnl::memory::data_type dt) {
    switch (dt) {
    case dnnl::memory::data_type::undef: 
        return "undef_data";
    case dnnl::memory::data_type::bf16: 
        return "ushort";
    case dnnl::memory::data_type::f16: 
        return "half";
    case dnnl::memory::data_type::f32: 
        return "float";
    case dnnl::memory::data_type::f64: 
        return "double";
    case dnnl::memory::data_type::s8: 
        return "char";
    case dnnl::memory::data_type::u8: 
        return "uchar";
    case dnnl::memory::data_type::f8_e4m3: 
        return "uchar";
    case dnnl::memory::data_type::f8_e5m2: 
        return "uchar";
    case dnnl::memory::data_type::f4_e2m1: 
        return "uchar";
    case dnnl::memory::data_type::f4_e3m0: 
        return "uchar";
    case dnnl::memory::data_type::e8m0: 
        return "uchar";
    case dnnl::memory::data_type::s4: 
        return "uchar";
    case dnnl::memory::data_type::u4: 
        return "uchar";
    case dnnl::memory::data_type::s32: 
        return "int";
    default:
        core::Error("Unexpected data type %d", int(dt));
        return "invalid";
    } 
}

} // namespace ocl
} // namespace onednn
} // namespace arhat

