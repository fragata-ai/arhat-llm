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

#include <cstdint>
#include <cassert>
#include <string>
#include <sstream>

#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

//
//    ShapeInfoArgs
//

ShapeInfoArgs::ShapeInfoArgs() { }

ShapeInfoArgs::~ShapeInfoArgs() { }

void ShapeInfoArgs::AddS32(const std::string &name, int64_t value) {
    m_data.push_back({Type::S32, name, value});
}

void ShapeInfoArgs::AddU32(const std::string &name, int64_t value) {
    m_data.push_back({Type::U32, name, value});
}

void ShapeInfoArgs::AddS64(const std::string &name, int64_t value) {
    m_data.push_back({Type::S64, name, value});
}

void ShapeInfoArgs::AddU64(const std::string &name, int64_t value) {
    m_data.push_back({Type::U64, name, value});
}

void ShapeInfoArgs::AddMemoryDesc(const std::string &name, const dnnl::memory::desc &md) {
    dnnl::memory::dim base = 0;
    dnnl::memory::dims dims(4, 0);
    dnnl::memory::dims strides(4, 0);
    if (!md.is_zero()) {
        base = md.get_submemory_offset();
        dims = md.get_dims();
        strides = md.get_strides();
    }
    std::string nameD = name + "_D";
    std::string nameS = name + "_S";
    AddU64(name + "_BASE", int64_t(base));
    AddS32(nameD + "0", int32_t(dims[0]));
    AddS32(nameD + "1", int32_t(dims[1]));
    AddS32(nameD + "2", int32_t(dims[2]));
    AddS32(nameD + "3", int32_t(dims[3]));
    AddS32(nameS + "0", int32_t(strides[0]));
    AddS32(nameS + "1", int32_t(strides[1]));
    AddS32(nameS + "2", int32_t(strides[2]));
    AddS32(nameS + "3", int32_t(strides[3]));
}

std::string ShapeInfoArgs::GetCode() {
    auto formatType = [](Type type) -> const char * {
        switch (type) {
        case Type::S32:
            return "int";
        case Type::U32:
            return "uint";
        case Type::S64:
            return "long";
        case Type::U64:
            return "ulong";
        default:
            assert(false);
            return "<error>";
        }
    };
    std::stringstream ss;
    for (const Entry &entry: m_data) {
        ss << ", const " << formatType(entry.type) << " " << entry.name;
    }
    return ss.str();
}

void ShapeInfoArgs::SetArgs(Kernel *kernel, int index) {
    for (const Entry &entry: m_data) {
        switch (entry.type) {
        case Type::S32:
            kernel->SetArgS32(index, int32_t(entry.value));
            break;
        case Type::U32:
            kernel->SetArgU32(index, uint32_t(entry.value));
            break;
        case Type::S64:
            kernel->SetArgS64(index, int64_t(entry.value));
            break;
        case Type::U64:
            kernel->SetArgU64(index, uint64_t(entry.value));
            break;
        default:
            assert(false);
            break;
        }
        index++;
    }
}

} // namespace ocl
} // namespace onednn
} // namespace arhat

