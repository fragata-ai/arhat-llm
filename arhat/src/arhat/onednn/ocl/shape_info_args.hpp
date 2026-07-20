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
#include <vector>

#include "dnnl.h"

#include "arhat/onednn/ocl/kernel.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

class ShapeInfoArgs {
public:
    ShapeInfoArgs();
    ~ShapeInfoArgs();
public:
    void AddS32(const std::string &name, int64_t value);
    void AddU32(const std::string &name, int64_t value);
    void AddS64(const std::string &name, int64_t value);
    void AddU64(const std::string &name, int64_t value);
    void AddMemoryDesc(const std::string &name, const dnnl::memory::desc &md);
    int Count() const {
        return int(m_data.size());
    }
    std::string GetCode();
    void SetArgs(Kernel *kernel, int index);
private:
    enum class Type {
        S32,
        U32,
        S64,
        U64
    };
private:
    struct Entry {
        Type type;
        std::string name;
        int64_t value;
    };
private:
    std::vector<Entry> m_data;
};

} // namespace ocl
} // namespace onednn
} // namespace arhat

