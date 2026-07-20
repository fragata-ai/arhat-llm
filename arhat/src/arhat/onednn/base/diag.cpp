/* 
* MIT License
*
* Copyright (c) 2020-2026 FRAGATA COMPUTER SYSTEMS AG
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
#include <vector>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"

namespace arhat {
namespace onednn {
namespace base {

namespace {

uint32_t F16ToBit32(uint16_t x) {
    uint32_t s = uint32_t(x & 0x8000) << 16;
    uint32_t e = uint32_t(x & 0x7c00) >> 10;
    uint32_t m = uint32_t(x & 0x03ff) << 13;
    if (e == 0x1f) {
        if (m == 0) {
            // Inf
            return (s | 0x7f800000 | m);
        } else {
            // NaN
            return (s | 0x7fc00000 | m);
        }
    }
    if (e == 0) {
        if (m == 0) {
            // zero
            return s;
        }
        // normalize
        e++;
        while ((m & 0x7f800000) == 0) {
            m <<= 1;
            e--;
        }
        m &= 0x007fffff;
    }
    return (s | ((e + (0x7f - 0xf)) << 23) | m);
}

float ToFloat(uint16_t x) {
    union {
        uint32_t i;
        float f;
    } u32;
    u32.i = F16ToBit32(x);
    return u32.f;
}

float ToFloat(float x) {
    return x;
}

float ToFloat(int32_t x) {
    return float(x);
}

struct MemoryStat {
    size_t volume = 0;
    double sum = 0.0;
    float fmin = 0.0f;
    size_t imin = 0;
    float fmax = 0.0f;
    size_t imax = 0;
};

template<typename T>
MemoryStat GetMemoryStat(const std::vector<T> &data) {
    MemoryStat stat;
    stat.volume = data.size();
    for (size_t i = 0; i < stat.volume; i++) {
        float x = ToFloat(data[i]);
        stat.sum += double(x);
        if (i == 0) {
            stat.fmin = x;
            stat.imin = 0;
            stat.fmax = x;
            stat.imax = 0;
        } else {
            if (x < stat.fmin) {
                stat.fmin = x;
                stat.imin = i;
            }
            if (x > stat.fmax) {
                stat.fmax = x;
                stat.imax = i;
            }
        }
    }
    return stat;
}

} // namespace

//
//    Diagnostics
//

void DiagPrintMemoryDims(const char *tag, const dnnl::memory::dims &dims) {
    printf("%s [", tag);
    int n = int(dims.size());
    for (int i = 0; i < n; i++) {
        if (i != 0) {
            printf(" ");
        }
        printf("%zd", size_t(dims[i]));
    }
    printf("]\n");
}

void DiagPrintMemoryDesc(const char *tag, const dnnl::memory::desc &desc) {
    dnnl::memory::dims dims = desc.get_dims();
    dnnl::memory::dims strides = desc.get_strides();
    dnnl::memory::dim offset = desc.get_submemory_offset();
    printf("%s\n", tag);
    DiagPrintMemoryDims("  dims", desc.get_dims());
    DiagPrintMemoryDims("  strides", desc.get_strides());
    if (desc.get_inner_nblks() != 0) {
        DiagPrintMemoryDims("  inner_blks", desc.get_inner_blks());
        DiagPrintMemoryDims("  inner_idxs", desc.get_inner_idxs());
    }
    if (offset != 0) {
        printf("  offset %zd\n", size_t(offset));
    }
}

void DiagPrintMemoryStat(const char *tag, NodeBase *node) {
    dnnl::memory::desc desc = node->MemoryDesc();
    dnnl::memory::data_type dt = node->MemoryType();
    if (dt != dnnl::memory::data_type::f16 && 
            dt != dnnl::memory::data_type::f32 &&
            dt != dnnl::memory::data_type::s32) {
        printf("%s: unsupported memory stat for data type %d\n", tag, int(dt));
        return;
    }
    dnnl::memory::dims dims = desc.get_dims();
    int ndims = desc.get_ndims();
    size_t volume = 1;
    for (int i = 0; i < ndims; i++) {
        volume *= size_t(dims[i]);
    }
    MemoryStat stat;
    if (dt == dnnl::memory::data_type::f16) {
        std::vector<uint16_t> data(volume);
        node->Read(data.data(), 0, int(volume * sizeof(uint16_t)));
        stat = GetMemoryStat(data);
    } else if (dt == dnnl::memory::data_type::f32) {
        std::vector<float> data(volume);
        node->Read(data.data(), 0, int(volume * sizeof(float)));
        stat = GetMemoryStat(data);        
    } else if (dt == dnnl::memory::data_type::s32) {
        std::vector<int32_t> data(volume);
        node->Read(data.data(), 0, int(volume * sizeof(int32_t)));
        stat = GetMemoryStat(data);        
    }
    printf("%s: volume %zd sum %g min %g (%zd) max %g (%zd)\n", 
        tag, stat.volume, stat.sum, stat.fmin, stat.imin, stat.fmax, stat.imax);
}

} // namespace base
} // namespace onednn
} // namespace arhat

