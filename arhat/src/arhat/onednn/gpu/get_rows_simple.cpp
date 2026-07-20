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

#include <string>
#include <memory>
#include <algorithm>
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
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/get_rows.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    GetRowsNode
//

class GetRowsNode: public NodeBase {
public:
    GetRowsNode(
        Context *context,
        core::Node *a,
        core::Node *b);
    ~GetRowsNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InferShapes();
    void InitArgs();
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    core::Node *m_a;
    core::Node *m_b;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_yDesc;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

GetRowsNode::GetRowsNode(
        Context *context,
        core::Node *a,
        core::Node *b):
            NodeBase(context),
            m_a(a),
            m_b(b) { }

GetRowsNode::~GetRowsNode() { }

bool GetRowsNode::Init() { 
    base::NodeBase *a = m_gpuContext->CastNode(m_a);
    base::NodeBase *b = m_gpuContext->CastNode(m_b);
    m_aDesc = a->MemoryDesc();
    m_bDesc = b->MemoryDesc();
    if (!Validate()) {
        return false;
    }
    m_aMem = a->Memory();
    m_bMem = b->Memory();
    InferShapes();
    SetMemory(m_yDesc);
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void GetRowsNode::Compute() {
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_memory);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 3);
    m_kernel->Launch(m_ndRange);
}

bool GetRowsNode::Validate() {
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    if (aType != dnnl::memory::data_type::f32 &&
            aType != dnnl::memory::data_type::f16 &&
            aType != dnnl::memory::data_type::bf16 &&
            aType != dnnl::memory::data_type::s32) {
        return false;
    }
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    if (bType != dnnl::memory::data_type::s32) {
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_aDesc) || !MemoryDescUtil::HasDenseRows(m_aDesc)) { 
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc)) { 
        return false;
    }
    return true;
}

void GetRowsNode::InferShapes() {
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims yDims{aDims[0], aDims[1], bDims[3], aDims[3]}; 
    // strange GGML rules
    dnnl::memory::data_type yType =
        (m_aDesc.get_data_type() == dnnl::memory::data_type::s32) ?
            dnnl::memory::data_type::s32 :
            dnnl::memory::data_type::f32;
    m_yDesc = dnnl::memory::desc(yDims, yType, dnnl::memory::format_tag::abcd);
}

void GetRowsNode::InitArgs() {
    dnnl::memory::dim aBase = m_aDesc.get_submemory_offset();
    dnnl::memory::dim bBase = m_bDesc.get_submemory_offset();
    dnnl::memory::dim yBase = m_yDesc.get_submemory_offset();
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims aStrides = m_aDesc.get_strides();
    dnnl::memory::dims bStrides = m_bDesc.get_strides();
    dnnl::memory::dims yStrides = m_yDesc.get_strides();

    uint32_t bDim1 = uint32_t(bDims[1]);
    uint32_t bDim1_fd0 = 0;
    uint32_t bDim1_fd1 = 0;
    ocl::MakeFastDiv(int64_t(bDim1), bDim1_fd0, bDim1_fd1); 

    m_shapeInfoArgs.AddS64("SRC0_BASE", bBase);
    m_shapeInfoArgs.AddS64("SRC1_BASE", aBase);
    m_shapeInfoArgs.AddS64("DST_BASE", yBase);
    m_shapeInfoArgs.AddS64("SRC0_D3", aDims[3]);
    m_shapeInfoArgs.AddU32("SRC1_D1", bDim1);
    m_shapeInfoArgs.AddU32("SRC1_D1_fd0", bDim1_fd0);
    m_shapeInfoArgs.AddU32("SRC1_D1_fd1", bDim1_fd1);
    m_shapeInfoArgs.AddS64("SRC1_D2", bDims[2]);
    m_shapeInfoArgs.AddS64("SRC0_S0", aStrides[0]);
    m_shapeInfoArgs.AddS64("SRC0_S1", aStrides[1]);
    m_shapeInfoArgs.AddS64("SRC0_S2", aStrides[2]);
    m_shapeInfoArgs.AddS64("SRC1_S1", bStrides[1]);
    m_shapeInfoArgs.AddS64("SRC1_S2", bStrides[2]);
    m_shapeInfoArgs.AddS64("SRC1_S3", bStrides[3]);
    m_shapeInfoArgs.AddS64("DST_S0", yStrides[0]);
    m_shapeInfoArgs.AddS64("DST_S1", yStrides[1]);
    m_shapeInfoArgs.AddS64("DST_S2", yStrides[2]);
}

void GetRowsNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = kernels::GetRowsSimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "get_rows_simple", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string GetRowsNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*get_rows_simple");
    sb.Int(int64_t(m_aDesc.get_data_type()));
    sb.Int(int64_t(m_yDesc.get_data_type()));
    return sb.Get();
}

std::string GetRowsNode::MakeProlog() {
    std::stringstream ss;
    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    dnnl::memory::data_type yType = m_yDesc.get_data_type();
    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitFastDiv(ss);
    ss << "#define SRC0_TYPE " << ocl::FormatType(aType) << "\n";
    ss << "#define DST_TYPE " << ocl::FormatType(yType) << "\n";
    ss << "\n";
    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";
    return ss.str();
}

void GetRowsNode::InitNdRange() {
    constexpr size_t BLOCK_SIZE = 256;          // TODO: Try also 64?
    constexpr size_t MAX_GWS = size_t(1) << 16; // TODO: What shall be correct limit here?
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    size_t blockNumY = ocl::DivUp(size_t(aDims[3]), BLOCK_SIZE);
    size_t lws0 = BLOCK_SIZE;
    size_t gws0 = size_t(bDims[3]) * lws0;
    size_t gws1 = std::min(size_t(blockNumY), MAX_GWS);
    size_t gws2 = std::min(size_t(bDims[1] * bDims[2]), MAX_GWS);
    m_ndRange = ocl::NdRange(gws0, gws1, gws2, lws0, 1, 1);
}

} // namespace

//
//    GetRowsSimple
//

GetRowsSimple::GetRowsSimple(Context *context):
        m_context(context) { }

GetRowsSimple::~GetRowsSimple() { }

std::unique_ptr<core::Node> GetRowsSimple::CreateNode(core::Node *a, core::Node *b) {
    std::unique_ptr<GetRowsNode> node = std::make_unique<GetRowsNode>(m_context, a, b);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

