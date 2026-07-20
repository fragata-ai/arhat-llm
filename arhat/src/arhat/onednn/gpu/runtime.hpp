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

#pragma once

#include <cstdint>
#include <string>
#include <array>
#include <memory>
#include <ostream>
#include <sstream>

#include "dnnl.hpp"

#include "arhat/onednn/base/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/device_info.hpp"

#include "arhat/onednn/gpu/kernel_cache.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    Context
//

class Context: public base::Context {
public:
    Context(base::Device *device);
    ~Context();
public:
    // node factories
    std::unique_ptr<core::Node> CreateAdd(
        core::Node *a, 
        core::Node *b,
        core::DataType dstType,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateSub(
        core::Node *a, 
        core::Node *b,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateMul(
        core::Node *a, 
        core::Node *b,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateDiv(
        core::Node *a, 
        core::Node *b,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateAddId(
        core::Node *a, 
        core::Node *b, 
        core::Node *ids) override;
    std::unique_ptr<core::Node> CreateSqr(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSqrt(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateLog(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSin(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateCos(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateCumSum(core::Node *a) override;
    std::unique_ptr<core::Node> CreateRepeat(core::Node *a, const core::Dims &shape) override;
    std::unique_ptr<core::Node> CreateNorm(
        core::Node *a, 
        float eps, 
        bool inplace) override;
    std::unique_ptr<core::Node> CreateRmsNorm(
        core::Node *a, 
        float eps, 
        bool inplace) override;
    std::unique_ptr<core::Node> CreateGroupNorm(
        core::Node *a, 
        int nGroups,
        float eps, 
        bool inplace) override;
    std::unique_ptr<core::Node> CreateL2Norm(
        core::Node *a, 
        float eps, 
        bool inplace) override;
    std::unique_ptr<core::Node> CreateMulMat(
        core::Node *a, 
        core::Node *b,
        core::Prec prec) override;
    std::unique_ptr<core::Node> CreateMulMatId(
        core::Node *a,
        core::Node *b,
        core::Node *ids) override;
    std::unique_ptr<core::Node> CreateOutProd(core::Node *a, core::Node *b) override;
    std::unique_ptr<core::Node> CreateSet(
        core::Node *a,
        core::Node *b,
        const core::Dims &stride,
        int offset,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateCpy(core::Node *a, core::Node *b) override;
    std::unique_ptr<core::Node> CreateGetRows(core::Node *a, core::Node *b) override;
    std::unique_ptr<core::Node> CreateSetRows(
        core::Node *a, 
        core::Node *b,
        core::Node *c) override;
    std::unique_ptr<core::Node> CreateDiag(core::Node *a) override;
    std::unique_ptr<core::Node> CreateRope(
        core::Node *a,
        core::Node *b,
        core::Node *c,
        int nDims,
        core::RopeMode mode,
        int nCtxOrig,
        float freqBase,
        float freqScale,
        float extFactor,
        float attnFactor,
        float betaFast,
        float betaSlow,
        const std::array<int, core::MropeSections> &sections, 
        bool inplace) override;
    std::unique_ptr<core::Node> CreatePad(
        core::Node *a,
        int lp0,
        int rp0,
        int lp1,
        int rp1,
        int lp2,
        int rp2,
        int lp3,
        int rp3,
        bool circular) override;
    std::unique_ptr<core::Node> CreateArgsort(core::Node *a, core::SortOrder order) override;
    std::unique_ptr<core::Node> CreateTri(core::Node *a, int mode) override;
    std::unique_ptr<core::Node> CreateFill(
        core::Node *a, 
        float value,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateFlashAttnExt(
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias,
        float logitSoftcap,
        core::Prec prec) override;
    std::unique_ptr<core::Node> CreateSsmConv(core::Node *sx, core::Node *c) override;
    std::unique_ptr<core::Node> CreateSolveTri(
        core::Node *a,
        core::Node *b,
        bool left,
        bool lower,
        bool uni) override;
    std::unique_ptr<core::Node> CreateGatedDeltaNet(
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *g,
        core::Node *beta,
        core::Node *state) override;
    // unary op node factories
    std::unique_ptr<core::Node> CreateAbs(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSgn(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateNeg(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateStep(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateTanh(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateElu(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateRelu(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSigmoid(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateGelu(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateGeluQuick(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSilu(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateHardswish(core::Node *a) override;
    std::unique_ptr<core::Node> CreateHardsigmoid(core::Node *a) override;
    std::unique_ptr<core::Node> CreateExp(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateExpm1(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSoftplus(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateGeluErf(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateXielu(
        core::Node *a,
        float alphaN,
        float alphaP,
        float beta,
        float eps) override;
    std::unique_ptr<core::Node> CreateFloor(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateCeil(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateRound(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateTrunc(core::Node *a, bool inplace) override;
    // GLU op node factories
    std::unique_ptr<core::Node> CreateReglu(
        core::Node *a, 
        core::Node *b,
        bool swapped) override;
    std::unique_ptr<core::Node> CreateGeglu(
        core::Node *a, 
        core::Node *b,
        bool swapped) override;
    std::unique_ptr<core::Node> CreateSwiglu(
        core::Node *a, 
        core::Node *b,
        bool swapped) override;
    std::unique_ptr<core::Node> CreateSwigluOai(
        core::Node *a, 
        core::Node *b,
        bool swapped,
        float alpha,
        float limit) override;
    std::unique_ptr<core::Node> CreateGegluErf(
        core::Node *a, 
        core::Node *b,
        bool swapped) override;
    std::unique_ptr<core::Node> CreateGegluQuick(
        core::Node *a, 
        core::Node *b,
        bool swapped) override;
public:
    ocl::OclContext *GetOclContext() {
        return m_oclContext.get();
    }
    ocl::DeviceInfo *GetDeviceInfo() {
        return m_deviceInfo.get();
    }
    void EnterKernel(
            const std::string &sig,
            const std::shared_ptr<ocl::Kernel> &kernel) {
        m_kernelCache.Enter(sig, kernel);
    }
    void EnterKernel(
            const std::string &sig,
            const std::shared_ptr<ocl::Kernel> &kernel,
            const ocl::NdRange &ndRange) {
        m_kernelCache.Enter(sig, kernel, ndRange);
    }
    bool FindKernel(
            const std::string &sig,
            std::shared_ptr<ocl::Kernel> &kernel) {
        return m_kernelCache.Find(sig, kernel);
    }
    bool FindKernel(
            const std::string &sig,
            std::shared_ptr<ocl::Kernel> &kernel,
            ocl::NdRange &ndRange) {
        return m_kernelCache.Find(sig, kernel, ndRange);
    }
private:
    void InitDeviceInfo();
private:
    std::unique_ptr<ocl::OclContext> m_oclContext;
    std::unique_ptr<ocl::DeviceInfo> m_deviceInfo;
    KernelCache m_kernelCache;
};

class NodeBase: public base::NodeBase {
public:
    NodeBase(Context *context);
    ~NodeBase();
public:
    Context *GetGpuContext() {
        return m_gpuContext;
    }
    ocl::OclContext *GetOclContext() {
        return m_oclContext;
    }
    void EnterKernel(
            const std::string &sig,
            const std::shared_ptr<ocl::Kernel> &kernel) {
        m_gpuContext->EnterKernel(sig, kernel);
    }
    void EnterKernel(
            const std::string &sig,
            const std::shared_ptr<ocl::Kernel> &kernel,
            const ocl::NdRange &ndRange) {
        m_gpuContext->EnterKernel(sig, kernel, ndRange);
    }
    bool FindKernel(
            const std::string &sig,
            std::shared_ptr<ocl::Kernel> &kernel) {
        return m_gpuContext->FindKernel(sig, kernel);
    }
    bool FindKernel(
            const std::string &sig,
            std::shared_ptr<ocl::Kernel> &kernel,
            ocl::NdRange &ndRange) {
        return m_gpuContext->FindKernel(sig, kernel, ndRange);
    }
    static void EmitDesc(
        std::ostream &os, 
        const std::string &name, 
        const dnnl::memory::desc &desc);
    static void EmitInt(
        std::ostream &os, 
        const std::string &name, 
        int64_t value);
protected:
    // GGML-specific - make this global?
    static constexpr int MaxNdims = 4;
protected:
    Context *m_gpuContext;
    ocl::OclContext *m_oclContext;
};

class NodeSigBuilder {
public:
    NodeSigBuilder();
    ~NodeSigBuilder();
public:
    void String(const std::string &value);
    void Bool(bool value);
    void Int(int64_t value);
    void Float(float value);
    void MemoryDesc(const dnnl::memory::desc &desc);
    void MemoryDims(const dnnl::memory::dims &dims);
    std::string Get();
private:
    void Sep();
private:
    std::stringstream m_ss;
    bool m_needSep;
};

} // namespace gpu
} // namespace onednn
} // namespace arhat

