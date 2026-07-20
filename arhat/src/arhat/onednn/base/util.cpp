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

#include <cstdint>
#include <cassert>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"
#include "arhat/onednn/base/quant.hpp"

namespace arhat {
namespace onednn {
namespace base {

//
//    Utilities
//

dnnl::memory::data_type MapDataType(core::DataType dataType) {
    switch (dataType) {
    case core::DataType::F16:
        return dnnl::memory::data_type::f16;
    case core::DataType::BF16:
        return dnnl::memory::data_type::bf16;
    case core::DataType::F32:
        return dnnl::memory::data_type::f32;
    case core::DataType::F64:
        return dnnl::memory::data_type::f64;
    case core::DataType::I32:
        return dnnl::memory::data_type::s32;
    case core::DataType::I8:
        return dnnl::memory::data_type::s8;
    case core::DataType::U8:
        return dnnl::memory::data_type::u8;
    case core::DataType::I64:
        // dirty hack to circumvent lack of I64 support in oneDNN
        // to be used with care in exceptional cases
        return dnnl::memory::data_type::f64;
    // quantized
    case core::DataType::Q2_K:
    case core::DataType::Q3_K:
    case core::DataType::Q4_0:
    case core::DataType::Q4_1:
    case core::DataType::Q4_K:
    case core::DataType::Q5_0:
    case core::DataType::Q5_1:
    case core::DataType::Q5_K:
    case core::DataType::Q6_K:
    case core::DataType::Q8_0:
    case core::DataType::Q8_1:
    case core::DataType::MXFP4:
        // treated as opaque byte arrays
        return dnnl::memory::data_type::u8;
    default:
        core::Error("Unsupported data type %d", int(dataType));
        return dnnl::memory::data_type::undef;
    }
}

dnnl::memory::dims MapDims(const core::Dims &dims) {
    return dnnl::memory::dims {
        dnnl::memory::dim(dims[3]),
        dnnl::memory::dim(dims[2]),
        dnnl::memory::dim(dims[1]),
        dnnl::memory::dim(dims[0])
    };
}

int64_t BytesToItems(dnnl::memory::data_type dataType, int64_t bytes) {
    switch (dataType) {
    case dnnl::memory::data_type::f4_e3m0:
        return bytes * 2;
    case dnnl::memory::data_type::f4_e2m1:
        return bytes * 2;
    case dnnl::memory::data_type::e8m0:
        return bytes;
    case dnnl::memory::data_type::f8_e5m2:
        return bytes;
    case dnnl::memory::data_type::f8_e4m3:
        return bytes;
    case dnnl::memory::data_type::f16:
        return (bytes + 1) / 2;
    case dnnl::memory::data_type::bf16:
        return (bytes + 1) / 2;
    case dnnl::memory::data_type::f32:
        return (bytes + 3) / 4;
    case dnnl::memory::data_type::f64:
        return (bytes + 7) / 8;
    case dnnl::memory::data_type::s32:
        return (bytes + 3) / 4;
    case dnnl::memory::data_type::s8:
        return bytes;
    case dnnl::memory::data_type::u8:
        return bytes;
    case dnnl::memory::data_type::s4:
        return bytes * 2;
    case dnnl::memory::data_type::u4:
        return bytes * 2;
    default:
        core::Error("Unsupported data type %d", int(dataType));
        return 0;
    }
}

int64_t ItemsToBytes(dnnl::memory::data_type dataType, int64_t items) {
    switch (dataType) {
    case dnnl::memory::data_type::f4_e3m0:
        return (items + 1) / 2;
    case dnnl::memory::data_type::f4_e2m1:
        return (items + 1) / 2;
    case dnnl::memory::data_type::e8m0:
        return items;
    case dnnl::memory::data_type::f8_e5m2:
        return items;
    case dnnl::memory::data_type::f8_e4m3:
        return items;
    case dnnl::memory::data_type::f16:
        return items * 2;
    case dnnl::memory::data_type::bf16:
        return items * 2;
    case dnnl::memory::data_type::f32:
        return items * 4;
    case dnnl::memory::data_type::f64:
        return items * 8;
    case dnnl::memory::data_type::s32:
        return items * 4;
    case dnnl::memory::data_type::s8:
        return items;
    case dnnl::memory::data_type::u8:
        return items;
    case dnnl::memory::data_type::s4:
        return (items + 1) / 2;
    case dnnl::memory::data_type::u4:
        return (items + 1) / 2;
    default:
        core::Error("Unsupported data type %d", int(dataType));
        return 0;
    }
}

dnnl::memory::format_tag DefaultFormatTag(int rank) {
    static dnnl::memory::format_tag tags[] = {
        dnnl::memory::format_tag::a,
        dnnl::memory::format_tag::ab,
        dnnl::memory::format_tag::abc,
        dnnl::memory::format_tag::abcd,
        dnnl::memory::format_tag::abcde,
        dnnl::memory::format_tag::abcdef
    };
    if (rank < 1 || rank > 6) {
        core::Error("Unsupported rank %d", rank);
    }
    return tags[rank - 1];
}

dnnl::memory::desc PlainMemoryDesc(const dnnl::memory::desc &desc) {
    return dnnl::memory::desc(
        desc.get_dims(), 
        desc.get_data_type(), 
        DefaultFormatTag(desc.get_ndims()));
}

bool IsRowMajor(const dnnl::memory::desc &desc) {
    if (desc.get_format_kind() != dnnl::memory::format_kind::blocked) {
        return false;
    }
    if (desc.get_inner_nblks() != 0) {
        return false;
    }
    int ndims = desc.get_ndims();
    dnnl::memory::dims dims = desc.get_dims();
    dnnl::memory::dims strides = desc.get_strides();
    dnnl::memory::dim s = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        if (strides[i] != s) {
            return false;
        }
        s *= dims[i];
    }
    return true;
}

int GetBlockSize(QuantMode quantMode) {
    switch (quantMode) {
    case QuantMode::Q2_K:
        return sizeof(Block_Q2_K);
    case QuantMode::Q3_K:
        return sizeof(Block_Q3_K);
    case QuantMode::Q4_0:
        return sizeof(Block_Q4_0);
    case QuantMode::Q4_1:
        return sizeof(Block_Q4_1);
    case QuantMode::Q4_K:
        return sizeof(Block_Q4_K);
    case QuantMode::Q5_0:
        return sizeof(Block_Q5_0);
    case QuantMode::Q5_1:
        return sizeof(Block_Q5_1);
    case QuantMode::Q5_K:
        return sizeof(Block_Q5_K);
    case QuantMode::Q6_K:
        return sizeof(Block_Q6_K);
    case QuantMode::Q8_0:
        return sizeof(Block_Q8_0);
    case QuantMode::Q8_1:
        return sizeof(Block_Q8_1);
    case QuantMode::MXFP4:
        return sizeof(Block_Mxfp4);
    default:
        assert(false);
        return 0;
    }
}

int GetQuantSize(QuantMode quantMode) {
    switch (quantMode) {
    case QuantMode::Q2_K:
        return QK_K;
    case QuantMode::Q3_K:
        return QK_K;
    case QuantMode::Q4_0:
        return QK4_0;
    case QuantMode::Q4_1:
        return QK4_1;
    case QuantMode::Q4_K:
        return QK_K;
    case QuantMode::Q5_0:
        return QK5_0;
    case QuantMode::Q5_1:
        return QK5_1;
    case QuantMode::Q5_K:
        return QK_K;
    case QuantMode::Q6_K:
        return QK_K;
    case QuantMode::Q8_0:
        return QK8_0;
    case QuantMode::Q8_1:
        return QK8_1;
    case QuantMode::MXFP4:
        return QK_MXFP4;
    default:
        assert(false);
        return 0;
    }
}

} // namespace base
} // namespace onednn
} // namespace arhat

