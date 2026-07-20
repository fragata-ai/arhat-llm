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
#include <memory>

#include "dnnl.hpp"

#include "arhat/onednn/base/runtime.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

class OclProgram;
class OclKernel;

//
//    OclDeviceInfo
//

struct OclDeviceInfo {
    size_t maxWorkGroupSize;
    size_t maxSlmBytesPerWg;
    bool supportsImmad;
};

//
//    OclContext
//

class OclContext {
public:
    OclContext() { }
    virtual ~OclContext() { }
public:
    static std::unique_ptr<OclContext> Create(base::Context *context);
public:
    virtual std::unique_ptr<OclProgram> 
        CreateProgram(
            int sourceCount,
            const char **source, 
            const char *options) = 0;
    virtual void EnqueueNDRangeKernel(
        OclKernel *kernel,
        int workDim,
        const size_t *globalWorkOffset,
        const size_t *globalWorkSize,
        const size_t *localWorkSize) = 0;
    virtual OclDeviceInfo GetDeviceInfo() = 0;
};

//
//    OclProgram
//

class OclProgram {
public:
    OclProgram() { }
    virtual ~OclProgram() { }
public:
    virtual std::unique_ptr<OclKernel> CreateKernel(const char *kernelName) = 0;
};

//
//    OclKernel
//

class OclKernel {
public:
    OclKernel() { }
    virtual ~OclKernel() { }
public:
    virtual int GetSpillMemSize() = 0;
    virtual void SetArgU8(int argIndex, uint8_t argValue) = 0;
    virtual void SetArgU16(int argIndex, uint16_t argValue) = 0;
    virtual void SetArgU32(int argIndex, uint32_t argValue) = 0;
    virtual void SetArgU64(int argIndex, uint64_t argValue) = 0;
    virtual void SetArgS8(int argIndex, int8_t argValue) = 0;
    virtual void SetArgS16(int argIndex, int16_t argValue) = 0;
    virtual void SetArgS32(int argIndex, int32_t argValue) = 0;
    virtual void SetArgS64(int argIndex, int64_t argValue) = 0;
    virtual void SetArgF32(int argIndex, float argValue) = 0;
    virtual void SetArgF64(int argIndex, double argValue) = 0;
    virtual void SetArgBuffer(int argIndex, const dnnl::memory &argValue) = 0;
};

} // namespace ocl
} // namespace onednn
} // namespace arhat

