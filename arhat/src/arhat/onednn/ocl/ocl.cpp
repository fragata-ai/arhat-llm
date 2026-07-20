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
#include <vector>
#include <memory>
#include <new>

#include <CL/opencl.h>

#include "dnnl.hpp"
#include "dnnl_ocl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"

#include "arhat/onednn/ocl/config_arch.hpp"
#include "arhat/onednn/ocl/ocl.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

namespace {

void CheckStatus(cl_int status) {
    if (status != CL_SUCCESS) {
        core::Error("OpenCL error: status %d", int(status));
    }
}

//
//    Implementation types
//

class OclProgramImpl;
class OclKernelImpl;

class OclContextImpl: public OclContext {
public:
    OclContextImpl(base::Context *context);
    ~OclContextImpl();
public:
    std::unique_ptr<OclProgram> CreateProgram(
        int sourceCount,
        const char **source, 
        const char *options) override;
    void EnqueueNDRangeKernel(
        OclKernel *kernel,
        int workDim,
        const size_t *globalWorkOffset,
        const size_t *globalWorkSize,
        const size_t *localWorkSize) override;
    OclDeviceInfo GetDeviceInfo() override;
private:
    void DumpProgramBuildLog(cl_program clProgram);
private:
    cl_device_id m_deviceId;
    cl_context m_context;
    cl_command_queue m_commandQueue;

};

class OclProgramImpl: public OclProgram {
public:
    OclProgramImpl(cl_device_id deviceId, cl_program program);
    ~OclProgramImpl();
public:
    std::unique_ptr<OclKernel> CreateKernel(const char *kernelName) override;
public:
    cl_program GetProgram() {
        return m_program;
    }
private:
    cl_device_id m_deviceId;
    cl_program m_program;
};

class OclKernelImpl: public OclKernel {
public:
    OclKernelImpl(cl_device_id deviceId, cl_kernel kernel);
    ~OclKernelImpl();
public:
    int GetSpillMemSize() override;
    void SetArgU8(int argIndex, uint8_t argValue) override;
    void SetArgU16(int argIndex, uint16_t argValue) override;
    void SetArgU32(int argIndex, uint32_t argValue) override;
    void SetArgU64(int argIndex, uint64_t argValue) override;
    void SetArgS8(int argIndex, int8_t argValue) override;
    void SetArgS16(int argIndex, int16_t argValue) override;
    void SetArgS32(int argIndex, int32_t argValue) override;
    void SetArgS64(int argIndex, int64_t argValue) override;
    void SetArgF32(int argIndex, float argValue) override;
    void SetArgF64(int argIndex, double argValue) override;
    void SetArgBuffer(int argIndex, const dnnl::memory &argValue) override;
public:
    cl_kernel GetKernel() {
        return m_kernel;
    }
private:
    cl_device_id m_deviceId;
    cl_kernel m_kernel;
};

//
//    OclContextImpl
//

OclContextImpl::OclContextImpl(base::Context *context):
        m_deviceId(nullptr),
        m_context(nullptr),
        m_commandQueue(nullptr) {
    m_deviceId = dnnl::ocl_interop::get_device(context->Engine());
    m_context = dnnl::ocl_interop::get_context(context->Engine());
    m_commandQueue = dnnl::ocl_interop::get_command_queue(context->Stream());
}

OclContextImpl::~OclContextImpl() { }

std::unique_ptr<OclProgram> OclContextImpl::CreateProgram(
        int sourceCount,
        const char **source, 
        const char *options) {
    cl_device_id devices[1] = {m_deviceId};
    cl_int status;
    cl_program clProgram = 
        clCreateProgramWithSource(m_context, sourceCount, source, nullptr, &status);
    CheckStatus(status);
    status = clBuildProgram(clProgram, 1, devices, options, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        DumpProgramBuildLog(clProgram);
    }
    CheckStatus(status);
    std::unique_ptr<OclProgramImpl> program;
    try {
        program = std::make_unique<OclProgramImpl>(m_deviceId, clProgram);
    } catch (const std::bad_alloc &) {
        clReleaseProgram(clProgram);
        throw;
    }
    return program;
}

void OclContextImpl::EnqueueNDRangeKernel(
        OclKernel *kernel,
        int workDim,
        const size_t *globalWorkOffset,
        const size_t *globalWorkSize,
        const size_t *localWorkSize) {
    OclKernelImpl *kernelImpl = static_cast<OclKernelImpl *>(kernel);
    cl_int status =
        clEnqueueNDRangeKernel(
            m_commandQueue,
            kernelImpl->GetKernel(),
            cl_uint(workDim),
            globalWorkOffset,
            globalWorkSize,
            localWorkSize,
            0,
            nullptr,
            nullptr);
    CheckStatus(status);
}

OclDeviceInfo OclContextImpl::GetDeviceInfo() {
    // Temporary solution (these properties should be queried at runtime)
    OclDeviceInfo info;
    if (CONFIG_GPU_ARCH == GPU_ARCH_UHD) {
        info.maxWorkGroupSize = 256;
        info.maxSlmBytesPerWg = 64 * 1024;
        info.supportsImmad = false;
    } else if (CONFIG_GPU_ARCH == GPU_ARCH_TGL) {
        info.maxWorkGroupSize = 512;
        info.maxSlmBytesPerWg = 64 * 1024;
        info.supportsImmad = false;
    } else if (CONFIG_GPU_ARCH == GPU_ARCH_ARL) {
        info.maxWorkGroupSize = 1024;
        info.maxSlmBytesPerWg = 64 * 1024;
        info.supportsImmad = false; // OpenVINO convention
    } else if (CONFIG_GPU_ARCH == GPU_ARCH_LNL) {
        info.maxWorkGroupSize = 1024;
        info.maxSlmBytesPerWg = 128 * 1024;
        info.supportsImmad = true;
    } else {
        assert(false);
    }
    return info;
}

void OclContextImpl::DumpProgramBuildLog(cl_program clProgram) {
    size_t size;
    cl_int status =
         clGetProgramBuildInfo(
            clProgram,
            m_deviceId,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &size);
    CheckStatus(status);
    std::vector<char> log(size);
    status =
        clGetProgramBuildInfo(
            clProgram,
            m_deviceId,
            CL_PROGRAM_BUILD_LOG,
            size,
            log.data(),
            nullptr);
    CheckStatus(status);
    fprintf(stderr, "%s", log.data());
}

//
//    OclProgramImpl
//

OclProgramImpl::OclProgramImpl(cl_device_id deviceId, cl_program program):
        m_deviceId(deviceId), m_program(program) { }

OclProgramImpl::~OclProgramImpl() {
    clReleaseProgram(m_program);
}

std::unique_ptr<OclKernel> OclProgramImpl::CreateKernel(const char *kernelName) {
    cl_int status;
    cl_kernel clKernel = clCreateKernel(m_program, kernelName, &status);
    CheckStatus(status);
    std::unique_ptr<OclKernelImpl> kernel;
    try {
        kernel = std::make_unique<OclKernelImpl>(m_deviceId, clKernel);
    } catch (const std::bad_alloc &) {
        clReleaseKernel(clKernel);
        throw;
    }
    return kernel;
}

//
//    OclKernelImpl
//

OclKernelImpl::OclKernelImpl(cl_device_id deviceId, cl_kernel kernel):
        m_deviceId(deviceId), m_kernel(kernel) { }

OclKernelImpl::~OclKernelImpl() { 
    clReleaseKernel(m_kernel);
}

int OclKernelImpl::GetSpillMemSize() {
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_required_subgroup_size.html
    cl_ulong value = 0;
    cl_int status = 
        clGetKernelWorkGroupInfo(
            m_kernel, 
            m_deviceId, 
            CL_KERNEL_SPILL_MEM_SIZE_INTEL,
            sizeof(value),
            &value,
            nullptr);
    return int(value);
}

void OclKernelImpl::SetArgU8(int argIndex, uint8_t argValue) {
    cl_uchar argTemp = cl_uchar(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgU16(int argIndex, uint16_t argValue) {
    cl_ushort argTemp = cl_ushort(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgU32(int argIndex, uint32_t argValue) {
    cl_uint argTemp = cl_uint(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgU64(int argIndex, uint64_t argValue) {
    cl_ulong argTemp = cl_ulong(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgS8(int argIndex, int8_t argValue) {
    cl_char argTemp = cl_char(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgS16(int argIndex, int16_t argValue) {
    cl_short argTemp = cl_short(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgS32(int argIndex, int32_t argValue) {
    cl_int argTemp = cl_int(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgS64(int argIndex, int64_t argValue) {
    cl_long argTemp = cl_long(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgF32(int argIndex, float argValue) {
    cl_float argTemp = cl_float(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgF64(int argIndex, double argValue) {
    cl_double argTemp = cl_double(argValue);
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

void OclKernelImpl::SetArgBuffer(int argIndex, const dnnl::memory &argValue) {
    cl_mem argTemp = argValue ? dnnl::ocl_interop::get_mem_object(argValue) : nullptr;
    cl_int status = clSetKernelArg(m_kernel, argIndex, sizeof(argTemp), &argTemp);
    CheckStatus(status);
}

} // namespace

//
//    OclContext
//

std::unique_ptr<OclContext> OclContext::Create(base::Context *context) {
    return std::make_unique<OclContextImpl>(context);
}

} // namespace ocl
} // namespace onednn
} // namespace arhat

