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

#include <cstdint>
#include <string>
#include <memory>
#include <ostream>
#include <sstream>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/mul_mat_id_helper.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    MulMatIdHelperImpl
//

class MulMatIdHelperImpl: public MulMatIdHelper {
public:
    MulMatIdHelperImpl(Context *context);
    ~MulMatIdHelperImpl();
public:
    bool Init(
        const dnnl::memory::desc &aDesc,
        const dnnl::memory::desc &bDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &idsADesc,
        const dnnl::memory::desc &idsCDesc,
        const dnnl::memory::desc &expertBoundsDesc);
    void Compute(
        const dnnl::memory &ids,
        const dnnl::memory &idsA,
        const dnnl::memory &idsC,
        const dnnl::memory &expertBounds) override;
private:
    void InitConfig();
    void InitArgs();
    bool Validate();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
    static void EmitInt(
        std::ostream &os, 
        const std::string &name, 
        int64_t value);
private:
    Context *m_context;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_idsDesc;
    dnnl::memory::desc m_idsADesc;
    dnnl::memory::desc m_idsCDesc;
    dnnl::memory::desc m_expertBoundsDesc;
    int32_t m_sgSize;
    int32_t m_nExpertUsed;
    int32_t m_neuPadded;
    int32_t m_neStore;
    int32_t m_nExperts;
    int32_t m_nTokens;
    int32_t m_nExpertUsedVar;
    int32_t m_nchannelsA;
    int32_t m_strideIds;
    int32_t m_strideIdsA;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

MulMatIdHelperImpl::MulMatIdHelperImpl(Context *context):
        m_context(context),
        m_sgSize(0),
        m_nExpertUsed(0),
        m_neuPadded(0),
        m_neStore(0),
        m_nExperts(0),
        m_nTokens(0),
        m_nExpertUsedVar(0),
        m_nchannelsA(0),
        m_strideIds(0),
        m_strideIdsA(0) { }

MulMatIdHelperImpl::~MulMatIdHelperImpl() { }

bool MulMatIdHelperImpl::Init(
        const dnnl::memory::desc &aDesc,
        const dnnl::memory::desc &bDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &idsADesc,
        const dnnl::memory::desc &idsCDesc,
        const dnnl::memory::desc &expertBoundsDesc) {
    m_aDesc = aDesc;
    m_bDesc = bDesc;
    m_idsDesc = idsDesc;
    m_idsADesc = idsADesc;
    m_idsCDesc = idsCDesc;
    m_expertBoundsDesc = expertBoundsDesc;
    InitConfig();
    InitArgs();
    if (!Validate()) {
        return false;
    }
    InitKernel();
    InitNdRange();
    return true;
}

void MulMatIdHelperImpl::Compute(
        const dnnl::memory &ids,
        const dnnl::memory &idsA,
        const dnnl::memory &idsC,
        const dnnl::memory &expertBounds) {
    m_kernel->SetArgBuffer(0, ids);
    m_kernel->SetArgBuffer(1, idsA);
    m_kernel->SetArgBuffer(2, idsC);
    m_kernel->SetArgBuffer(3, expertBounds);
    m_kernel->SetArgS32(4, m_nTokens);
    m_kernel->SetArgS32(5, m_nExpertUsedVar);
    m_kernel->SetArgS32(6, m_nchannelsA);
    m_kernel->SetArgS32(7, m_strideIds);
    m_kernel->SetArgS32(8, m_strideIdsA);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 9);
    m_kernel->Launch(m_ndRange);
}

void MulMatIdHelperImpl::InitConfig() {
    ocl::OclDeviceInfo info = m_context->GetOclContext()->GetDeviceInfo();
    m_sgSize = 32;
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    dnnl::memory::dim nExpertUsed = idsDims[3];
    switch (nExpertUsed) {
    case 2:
    case 4:
    case 6:
    case 8:
    case 16:
    case 32:
        m_nExpertUsed = int32_t(nExpertUsed);
        break;
    default:
        m_nExpertUsed = 0;
        break;
    }
    m_neuPadded = (m_nExpertUsed == 6) ? 8 : m_nExpertUsed;
    // neStore = nTokens is variable - for simpicity, just request all available SLM
    m_neStore = int32_t(info.maxSlmBytesPerWg / sizeof(int32_t)); 
}

void MulMatIdHelperImpl::InitArgs() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims idsDims = m_idsDesc.get_dims();
    dnnl::memory::dims aStrides = m_aDesc.get_strides();
    dnnl::memory::dims idsStrides = m_idsDesc.get_strides();

    m_nExperts = int32_t(bDims[1]);
    m_nTokens = int32_t(aDims[1]);
    m_nExpertUsedVar = int32_t(idsDims[3]);
    m_nchannelsA = int32_t(aDims[2]);
    m_strideIds = int32_t(idsStrides[2]);
    m_strideIdsA = int32_t(aStrides[1] / aStrides[2]);

    size_t idsBase = m_idsDesc.get_submemory_offset();
    size_t idsABase = m_idsADesc.get_submemory_offset();
    size_t idsCBase = m_idsCDesc.get_submemory_offset();
    size_t expertBoundsBase = m_expertBoundsDesc.get_submemory_offset();
    m_shapeInfoArgs.AddS64("IDS_BASE", idsBase);
    m_shapeInfoArgs.AddS64("IDS_SRC1_BASE", idsABase);
    m_shapeInfoArgs.AddS64("IDS_DST_BASE", idsCBase);
    m_shapeInfoArgs.AddS64("EXPERT_BOUNDS_BASE", expertBoundsBase);
}

bool MulMatIdHelperImpl::Validate() {
    // store constraints
    if (m_nTokens >= (1 << 22) || m_nExpertUsedVar >= (1 << 10)) {
        return false;
    }
    // SLM constraints
    if (m_nTokens * sizeof(int32_t) > m_neStore) {
        return false;
    }
    return true;
}

void MulMatIdHelperImpl::InitKernel() {
    std::string sig = MakeSig();
    if (m_context->FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::MulMatIdHelperKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_context->GetOclContext(), 
        "mm_ids_helper", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    m_context->EnterKernel(sig, m_kernel);
}

std::string MulMatIdHelperImpl::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*mm_ids_helper");
    sb.Int(int64_t(m_nExpertUsed));
    return sb.Get();
}

std::string MulMatIdHelperImpl::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitUnroll(ss);
    EmitInt(ss, "SG_SIZE", m_sgSize);
    EmitInt(ss, "N_EXPERT_USED", m_nExpertUsed);
    EmitInt(ss, "NEU_PADDED", m_neuPadded);
    EmitInt(ss, "NE_STORE", m_neStore);
    ss << "\n";
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void MulMatIdHelperImpl::InitNdRange() {
    size_t lws0 = size_t(m_sgSize);
    size_t gws0 = size_t(m_nExperts) * lws0;
    m_ndRange = ocl::NdRange(gws0, 1, 1, lws0, 1, 1);
}

void MulMatIdHelperImpl::EmitInt(
        std::ostream &os, 
        const std::string &name, 
        int64_t value) {
    os << "#define " << name << " " << ocl::FormatInt(value) << "\n";
}

} // namespace

std::unique_ptr<MulMatIdHelper> CreateMulMatIdHelper(
        Context *context,
        const dnnl::memory::desc &aDesc,
        const dnnl::memory::desc &bDesc,
        const dnnl::memory::desc &idsDesc,
        const dnnl::memory::desc &idsADesc,
        const dnnl::memory::desc &idsCDesc,
        const dnnl::memory::desc &expertBoundsDesc) {
    std::unique_ptr<MulMatIdHelperImpl> h = std::make_unique<MulMatIdHelperImpl>(context);
    if (!h->Init(
            aDesc, 
            bDesc, 
            idsDesc, 
            idsADesc, 
            idsCDesc, 
            expertBoundsDesc)) {
        return nullptr;
    }
    return h;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

