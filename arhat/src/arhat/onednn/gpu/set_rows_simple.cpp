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
#include <cassert>
#include <string>
#include <memory>
#include <ostream>
#include <sstream>
#include <limits>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
#include "arhat/onednn/gpu/set_rows.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    SetRowsNode
//

class SetRowsNode: public NodeBase {
public:
    SetRowsNode(
        Context *context,
        core::Node *a,
        core::Node *b,
        core::Node *c);
    ~SetRowsNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InitArgs();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    core::Node *m_a;
    core::Node *m_b;
    core::Node *m_c;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_cDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    dnnl::memory m_cMem;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

SetRowsNode::SetRowsNode(
        Context *context,
        core::Node *a,
        core::Node *b,
        core::Node *c):
            NodeBase(context),
            m_a(a),
            m_b(b),
            m_c(c) { }

SetRowsNode::~SetRowsNode() { }

bool SetRowsNode::Init() { 
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    base::NodeBase *c = m_gpuContext->CastNode(m_c);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    m_cDesc = c->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    m_cMem = c->Memory();
    SetMemory(m_aDesc, m_aMem);
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

bool SetRowsNode::Validate() {
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    if (aType != dnnl::memory::data_type::f32 &&
            aType != dnnl::memory::data_type::f16 &&
            aType != dnnl::memory::data_type::bf16) {
        return false;
    }
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    if (bType != dnnl::memory::data_type::f32) {
        return false;
    }
    dnnl::memory::data_type cType = m_cDesc.get_data_type();
    if (cType != dnnl::memory::data_type::s32 &&
            cType != dnnl::memory::data_type::f64) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) || 
            !MemoryDescUtil::HasDenseRows(m_aDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc) || 
            !MemoryDescUtil::HasDenseRows(m_bDesc)) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_cDesc)) {
        return false;
    }
    return true;
}

void SetRowsNode::Compute() {
    // note weird order (b, c, a)
    m_kernel->SetArgBuffer(0, m_bMem);
    m_kernel->SetArgBuffer(1, m_cMem);
    m_kernel->SetArgBuffer(2, m_aMem);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 3);
    m_kernel->Launch(m_ndRange);
}

void SetRowsNode::InitArgs() {
    dnnl::memory::dim aBase = m_aDesc.get_submemory_offset();
    dnnl::memory::dim bBase = m_bDesc.get_submemory_offset();
    dnnl::memory::dim cBase = m_cDesc.get_submemory_offset();
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    dnnl::memory::dims aStrides = m_aDesc.get_strides();
    dnnl::memory::dims bStrides = m_bDesc.get_strides();
    dnnl::memory::dims cStrides = m_cDesc.get_strides();

    // SRC0 = b, SRC1 = c, DST = a
    m_shapeInfoArgs.AddS64("SRC0_BASE", bBase);
    m_shapeInfoArgs.AddS64("SRC1_BASE", cBase);
    m_shapeInfoArgs.AddS64("DST_BASE", aBase);
    m_shapeInfoArgs.AddS32("SRC0_D2", bDims[2]);
    m_shapeInfoArgs.AddS32("SRC0_S0", bStrides[0]);
    m_shapeInfoArgs.AddS32("SRC0_S1", bStrides[1]);
    m_shapeInfoArgs.AddS32("SRC0_S2", bStrides[2]);
    m_shapeInfoArgs.AddS32("SRC1_D1", cDims[1]);
    m_shapeInfoArgs.AddS32("SRC1_D2", cDims[2]);
    m_shapeInfoArgs.AddS32("SRC1_S1", cStrides[1]);
    m_shapeInfoArgs.AddS32("SRC1_S2", cStrides[2]);
    m_shapeInfoArgs.AddS32("SRC1_S3", cStrides[3]);
    m_shapeInfoArgs.AddS32("DST_D3", aDims[3]);
    m_shapeInfoArgs.AddS32("DST_S0", aStrides[0]);
    m_shapeInfoArgs.AddS32("DST_S1", aStrides[1]);
    m_shapeInfoArgs.AddS32("DST_S2", aStrides[2]);
}

void SetRowsNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::SetRowsSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "set_rows_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string SetRowsNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*set_rows_simple");
    sb.Int(int64_t(m_aDesc.get_data_type()));
    sb.Int(int64_t(m_bDesc.get_data_type()));
    sb.Int(int64_t(m_cDesc.get_data_type()));
    return sb.Get();
}

std::string SetRowsNode::MakeProlog() {
    std::stringstream ss;
    ocl::CommonXe::EmitGrid(ss);
    // note weird order (b, c, a)
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    dnnl::memory::data_type cType = m_cDesc.get_data_type();
    ss << "#define SRC0_TYPE " << ocl::FormatType(bType) << "\n";
    if (cType == dnnl::memory::data_type::f64) {
        // dirty hack to circumvent lack of int64 support in oneDNN
        ss << "#define SRC1_TYPE long\n";
    } else {
        ss << "#define SRC1_TYPE " << ocl::FormatType(cType) << "\n";
    }
    ss << "#define DST_TYPE " << ocl::FormatType(aType) << "\n";
    ss << "\n";
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void SetRowsNode::InitNdRange() {
    ocl::OclDeviceInfo info = GetOclContext()->GetDeviceInfo();
    // a is DST, b is SRC0
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    size_t rowSize = size_t(aDims[3]);
    size_t lws0 = 32;
    if (lws0 < rowSize && lws0 < info.maxWorkGroupSize) {
        lws0 *= 2;
    }
    size_t rowsPerWg = 1;
    if (lws0 > rowSize) {
        // ACHTUNG: What if not divisible?
        rowsPerWg = lws0 / rowSize;
        lws0 = rowSize;
    }
    size_t gws0 = (size_t(bDims[2] + rowsPerWg - 1) / rowsPerWg) * lws0;
    size_t gws1 = size_t(bDims[1]) * rowsPerWg;
    size_t gws2 = size_t(bDims[0]);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, rowsPerWg, 1);
}

} // namespace

//
//    SetRowsSimple
//

SetRowsSimple::SetRowsSimple(Context *context):
        m_context(context) { }

SetRowsSimple::~SetRowsSimple() { }

std::unique_ptr<core::Node> SetRowsSimple::CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *c) {
    std::unique_ptr<SetRowsNode> node = std::make_unique<SetRowsNode>(m_context, a, b, c);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

