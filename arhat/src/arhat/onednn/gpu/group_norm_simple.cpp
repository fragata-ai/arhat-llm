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
#include <algorithm>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/group_norm.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    GroupNormNode
//

class GroupNormNode: public NodeBase {
public:
    GroupNormNode(
        Context *context,
        core::Node *a, 
        int nGroups,
        float eps, 
        bool inplace);
    ~GroupNormNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InitConfig();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    core::Node *m_a; 
    int m_nGroups;
    float m_eps;
    bool m_inplace;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_aMem;
    int64_t m_groupSize;
    int64_t m_sgSize;
    int64_t m_sgNum;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

GroupNormNode::GroupNormNode(
        Context *context,
        core::Node *a, 
        int nGroups,
        float eps, 
        bool inplace):
            NodeBase(context),
            m_a(a),
            m_nGroups(nGroups),
            m_eps(eps),
            m_inplace(inplace),
            m_groupSize(0), 
            m_sgSize(0), 
            m_sgNum(0) { }

GroupNormNode::~GroupNormNode() { }

bool GroupNormNode::Init() { 
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    m_aDesc = a->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_yDesc = m_aDesc;
    if (m_inplace) {
        SetMemory(m_yDesc, m_aMem);
    } else {
        SetMemory(m_yDesc);
    }
    InitConfig();
    InitKernel();
    return true;
}

void GroupNormNode::Compute() { 
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_memory);
    m_kernel->SetArgF32(2, m_eps);
    m_kernel->Launch(m_ndRange);
}

bool GroupNormNode::Validate() {
    if (m_aDesc.get_data_type() != dnnl::memory::data_type::f32) {
        return false;
    }
    if (!base::IsRowMajor(m_aDesc)) {
        return false;
    }
    dnnl::memory::dims dims = m_aDesc.get_dims();
    dnnl::memory::dims strides = m_aDesc.get_strides();
    dnnl::memory::dim s3 = 1;
    dnnl::memory::dim s2 = s3 * dims[3];
    dnnl::memory::dim s1 = s2 * dims[2];
    dnnl::memory::dim s0 = s1 * dims[1];
    if (strides[0] != s0 || strides[1] != s1 || strides[2] != s2 || strides[3] != s3) {
        return false;
    }
    return true;
}

void GroupNormNode::InitConfig() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    m_groupSize = 
        int64_t(aDims[3]) * 
        int64_t(aDims[2]) * 
        ((int64_t(aDims[1]) + int64_t(m_nGroups) - 1) / int64_t(m_nGroups));
    m_sgSize = 32; // TODO: Use device min subgroup size
    m_sgNum = m_groupSize;
}

void GroupNormNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel, m_ndRange)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::GroupNormSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "group_norm_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    InitNdRange();
    EnterKernel(sig, m_kernel, m_ndRange);
}

std::string GroupNormNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("group_norm_simple");
    sb.MemoryDesc(m_aDesc);
    sb.Int(int64_t(m_nGroups));
    sb.Float(m_eps);
    sb.Bool(m_inplace);
    return sb.Get();
}

std::string GroupNormNode::MakeProlog() {
    std::stringstream ss;
    EmitInt(ss, "GROUP_SIZE", m_groupSize);
    EmitInt(ss, "SG_SIZE", m_sgSize);
    EmitInt(ss, "SG_NUM", m_sgNum);
    ss << "\n";
    EmitDesc(ss, "SRC", m_aDesc);
    EmitDesc(ss, "DST", m_yDesc);
    return ss.str();
}

void GroupNormNode::InitNdRange() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    size_t gws0 = size_t(int64_t(m_nGroups) * m_sgSize);
    size_t lws0 = size_t(m_sgSize);
    m_ndRange = ocl::NdRange(gws0, 1, 1, lws0, 1, 1);
}

} // namespace

//
//    GroupNormSimple
//

GroupNormSimple::GroupNormSimple(Context *context):
        m_context(context) { }

GroupNormSimple::~GroupNormSimple() { }

std::unique_ptr<core::Node> GroupNormSimple::CreateNode(
        core::Node *a, 
        int nGroups,
        float eps, 
        bool inplace) {
    std::unique_ptr<GroupNormNode> node = 
        std::make_unique<GroupNormNode>(m_context, a, nGroups, eps, inplace);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

