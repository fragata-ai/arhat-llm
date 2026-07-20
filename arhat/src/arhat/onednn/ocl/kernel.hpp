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
#include <cassert>
#include <string>
#include <array>
#include <set>
#include <map>
#include <memory>

#include "dnnl.hpp"

#include "arhat/onednn/ocl/ocl.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

//
//    NdRange
//

class NdRange {
public:
    NdRange();
    NdRange(size_t gws0, size_t lws0);
    NdRange(
        size_t gws0, 
        size_t gws1, 
        size_t lws0, 
        size_t lws1);
    NdRange(
        size_t gws0, 
        size_t gws1, 
        size_t gws2, 
        size_t lws0, 
        size_t lws1, 
        size_t lws2);
    ~NdRange();
public:
    int Ndims() const {
        return m_ndims;
    }
    const size_t *Gws() const {
        return m_gws.data();
    }
    const size_t *Lws() const {
        return m_lws.data();
    }
private:
    int m_ndims;
    std::array<size_t, 3> m_gws;
    std::array<size_t, 3> m_lws;
};

//
//    KernelContext
//

class KernelContext {
public:
    KernelContext();
    ~KernelContext();
public:
    void SetOption(const std::string &option);
    void SetStrVar(const std::string &name, const std::string &value);
    void SetIntVar(const std::string name, int64_t value);
    void SetFloatVar(const std::string &name, float value);
    std::string Format() const;
private:
    void SetDefaults();
    void CheckVarName(const std::string &name);
private:
    std::set<std::string> m_options;
    std::map<std::string, std::string> m_strVars;
    std::map<std::string, int64_t> m_intVars;
    std::map<std::string, float> m_floatVars;
};

//
//    Kernel
//

class Kernel {
public:
    Kernel();
    ~Kernel();
public:
    void Init(
        OclContext *oclContext,
        const std::string &name,
        const KernelContext &kernelContext,
        const char *prolog,
        const char *code);
    int GetSpillMemSize();
    void SetArgU8(int index, uint8_t value);
    void SetArgU16(int index, uint16_t value);
    void SetArgU32(int index, uint32_t value);
    void SetArgU64(int index, uint64_t value);
    void SetArgS8(int index, int8_t value);
    void SetArgS16(int index, int16_t value);
    void SetArgS32(int index, int32_t value);
    void SetArgS64(int index, int64_t value);
    void SetArgF32(int index, float value);
    void SetArgF64(int index, double value);
    void SetArgBuffer(int index, const dnnl::memory &value);
    void Launch(const NdRange &range);
private:
    OclContext *m_oclContext;
    std::unique_ptr<OclProgram> m_oclProgram;
    std::unique_ptr<OclKernel> m_oclKernel;
};

} // namespace ocl
} // namespace onednn
} // namespace arhat

