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
#include <algorithm>

#include "arhat/core/runtime.hpp" 

#include "arhat/onednn/ocl/ocl.hpp"
#include "arhat/onednn/ocl/kernel.hpp"
#include "arhat/onednn/ocl/common_xe.hpp"
#include "arhat/onednn/ocl/shape_info_args.hpp"
#include "arhat/onednn/ocl/util.hpp"

#include "arhat/onednn/kernels/code.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/binary.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

namespace {

//
//    BinaryNode
//

class BinaryNode: public NodeBase {
public:
    BinaryNode(
        Context *context,
        BinaryOp op,
        core::Node *a, 
        core::Node *b,
        core::DataType dstType,
        bool inplace);
    ~BinaryNode();
public:
    bool Init();
public:
    void Compute() override;
private:
    bool Validate();
    void InferShapes();
    void InitArgs();
    bool MustCollapse();
    static void Collapse(dnnl::memory::dims &dims, dnnl::memory::dims &strides);
    void InitGrid(const dnnl::memory::dims &cDims);
    void InitKernel();
    std::string MakeSig();
    std::string MakeProlog();
    void InitNdRange();
private:
    BinaryOp m_op;
    core::Node *m_a;
    core::Node *m_b;
    core::DataType m_dstType;
    bool m_inplace;
    dnnl::memory::desc m_aDesc;
    dnnl::memory::desc m_bDesc;
    dnnl::memory::desc m_cDesc;
    bool m_isNop;
    dnnl::memory m_aMem;
    dnnl::memory m_bMem;
    size_t m_lws[3];
    size_t m_gws[3];
    bool m_unravel;
    ocl::ShapeInfoArgs m_shapeInfoArgs;
    std::shared_ptr<ocl::Kernel> m_kernel;
    ocl::NdRange m_ndRange;
};

BinaryNode::BinaryNode(
        Context *context,
        BinaryOp op,
        core::Node *a, 
        core::Node *b,
        core::DataType dstType,
        bool inplace):
            NodeBase(context),
            m_op(op),
            m_a(a),
            m_b(b),
            m_dstType(dstType),
            m_inplace(inplace),
            m_isNop(false),
            m_lws{0, 0, 0},
            m_gws{0, 0, 0},
            m_unravel(false) { }

BinaryNode::~BinaryNode() { }

bool BinaryNode::Init() {
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
    if (m_isNop) {
        return true;
    }
    InitArgs();
    InitKernel();
    InitNdRange();
    return true;
}

void BinaryNode::Compute() {
    if (m_isNop) {
        return;
    }
    m_kernel->SetArgBuffer(0, m_aMem);
    m_kernel->SetArgBuffer(1, m_bMem);
    m_kernel->SetArgBuffer(2, m_memory);
    m_shapeInfoArgs.SetArgs(m_kernel.get(), 3);
    m_kernel->Launch(m_ndRange);
}

bool BinaryNode::Validate() {
    assert(m_aDesc.get_ndims() == 4);
    assert(m_bDesc.get_ndims() == 4);

    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    dnnl::memory::data_type bType = m_bDesc.get_data_type();

    if (aType != dnnl::memory::data_type::f32 && aType != dnnl::memory::data_type::f16) {
        return false;
    }
    if (bType != dnnl::memory::data_type::f32 && bType != dnnl::memory::data_type::f16) {
        return false;
    }
    if (m_dstType != core::DataType::Undef &&
            m_dstType != core::DataType::F32 &&
            m_dstType != core::DataType::F16) {
        return false;
    }

    if (!MemoryDescUtil::IsPlain(m_aDesc)) { 
        return false;
    }
    if (!MemoryDescUtil::IsPlain(m_bDesc)) { 
        return false;
    }

    return true;
}

void BinaryNode::InferShapes() {
    dnnl::memory::data_type cType = 
        (m_dstType != core::DataType::Undef) ?
            base::MapDataType(m_dstType) : m_aDesc.get_data_type();
    if (m_inplace) {
        assert(cType == m_aDesc.get_data_type());
        m_cDesc = m_aDesc;
        SetMemory(m_cDesc, m_aMem);
    } else {
        m_cDesc =
            dnnl::memory::desc(
                m_aDesc.get_dims(), 
                cType, 
                base::DefaultFormatTag(m_aDesc.get_ndims()));
        SetMemory(m_cDesc);
    }
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    m_isNop = (cDims[0] * cDims[1] * cDims[2] * cDims[3] == 0);
}

void BinaryNode::InitArgs() {
    dnnl::memory::dim aBase = m_aDesc.get_submemory_offset();
    dnnl::memory::dim bBase = m_bDesc.get_submemory_offset();
    dnnl::memory::dim cBase = m_cDesc.get_submemory_offset();
    dnnl::memory::dims aDims = m_aDesc.get_dims();
    dnnl::memory::dims aStrides = m_aDesc.get_strides();
    dnnl::memory::dims bDims = m_bDesc.get_dims();
    dnnl::memory::dims bStrides = m_bDesc.get_strides();
    dnnl::memory::dims cDims = m_cDesc.get_dims();
    dnnl::memory::dims cStrides = m_cDesc.get_strides();

    if (MustCollapse()) {
        bool bcast[4];
        for (int i = 0; i < 4; i++) {
            bcast[i] = (bDims[i] != cDims[i]);
        }
        for (int i = 3; i >= 0; i--) {
            if (bcast[i]) {
                break;
            }
            if (i < 3) {
                Collapse(aDims, aStrides);
                Collapse(bDims, bStrides);
                Collapse(cDims, cStrides);
            }
        }
    }

    InitGrid(cDims);

    m_shapeInfoArgs.AddS64("SRC0_BASE", aBase);
    m_shapeInfoArgs.AddS64("SRC1_BASE", bBase);
    m_shapeInfoArgs.AddS64("DST_BASE", cBase);

    if (!m_unravel) {
        uint32_t cDim0 = int32_t(cDims[0]);
        int32_t cDim1 = int32_t(cDims[1]);
        int32_t cDim2 = int32_t(cDims[2]);
        int32_t cDim3 = int32_t(cDims[3]);
        uint32_t bDim0 = uint32_t(bDims[0]);
        uint32_t bDim1 = uint32_t(bDims[1]);
        uint32_t bDim2 = uint32_t(bDims[2]);
        uint32_t bDim3 = uint32_t(bDims[3]);
        int32_t cStride0 = int32_t(cStrides[0]);
        int32_t cStride1 = int32_t(cStrides[1]);
        int32_t cStride2 = int32_t(cStrides[2]);
        int32_t aStride0 = int32_t(aStrides[0]);
        int32_t aStride1 = int32_t(aStrides[1]);
        int32_t aStride2 = int32_t(aStrides[2]);
        int32_t aStride3 = int32_t(aStrides[3]);
        int32_t bStride0 = int32_t(bStrides[0]);
        int32_t bStride1 = int32_t(bStrides[1]);
        int32_t bStride2 = int32_t(bStrides[2]);
        int32_t bStride3 = int32_t(bStrides[3]);

        uint32_t cDim0_fd0, cDim0_fd1;
        ocl::MakeFastDiv(int64_t(cDim0), cDim0_fd0, cDim0_fd1);
        uint32_t bDim0_fd0, bDim0_fd1;
        ocl::MakeFastDiv(int64_t(bDim0), bDim0_fd0, bDim0_fd1);
        uint32_t bDim1_fd0, bDim1_fd1;
        ocl::MakeFastDiv(int64_t(bDim1), bDim1_fd0, bDim1_fd1);
        uint32_t bDim2_fd0, bDim2_fd1;
        ocl::MakeFastDiv(int64_t(bDim2), bDim2_fd0, bDim2_fd1);
        uint32_t bDim3_fd0, bDim3_fd1;
        ocl::MakeFastDiv(int64_t(bDim3), bDim3_fd0, bDim3_fd1);

        m_shapeInfoArgs.AddU32("DST_D0", cDim0);
        m_shapeInfoArgs.AddU32("DST_D0_fd0", cDim0_fd0);
        m_shapeInfoArgs.AddU32("DST_D0_fd1", cDim0_fd1);
        m_shapeInfoArgs.AddS32("DST_D1", cDim1);
        m_shapeInfoArgs.AddS32("DST_D2", cDim2);
        m_shapeInfoArgs.AddS32("DST_D3", cDim3);
        m_shapeInfoArgs.AddU32("SRC1_D0", bDim0);
        m_shapeInfoArgs.AddU32("SRC1_D0_fd0", bDim0_fd0);
        m_shapeInfoArgs.AddU32("SRC1_D0_fd1", bDim0_fd1);
        m_shapeInfoArgs.AddU32("SRC1_D1", bDim1);
        m_shapeInfoArgs.AddU32("SRC1_D1_fd0", bDim1_fd0);
        m_shapeInfoArgs.AddU32("SRC1_D1_fd1", bDim1_fd1);
        m_shapeInfoArgs.AddU32("SRC1_D2", bDim2);
        m_shapeInfoArgs.AddU32("SRC1_D2_fd0", bDim2_fd0);
        m_shapeInfoArgs.AddU32("SRC1_D2_fd1", bDim2_fd1);
        m_shapeInfoArgs.AddU32("SRC1_D3", bDim3);
        m_shapeInfoArgs.AddU32("SRC1_D3_fd0", bDim3_fd0);
        m_shapeInfoArgs.AddU32("SRC1_D3_fd1", bDim3_fd1);
        m_shapeInfoArgs.AddS32("DST_S0", cStride0);
        m_shapeInfoArgs.AddS32("DST_S1", cStride1);
        m_shapeInfoArgs.AddS32("DST_S2", cStride2);
        m_shapeInfoArgs.AddS32("SRC0_S0", aStride0);
        m_shapeInfoArgs.AddS32("SRC0_S1", aStride1);
        m_shapeInfoArgs.AddS32("SRC0_S2", aStride2);
        m_shapeInfoArgs.AddS32("SRC0_S3", aStride3);
        m_shapeInfoArgs.AddS32("SRC1_S0", bStride0);
        m_shapeInfoArgs.AddS32("SRC1_S1", bStride1);
        m_shapeInfoArgs.AddS32("SRC1_S2", bStride2);
        m_shapeInfoArgs.AddS32("SRC1_S3", bStride3);

    } else {
        uint32_t cDim0 = uint32_t(cDims[0]);
        uint32_t cDim1 = uint32_t(cDims[1]);
        uint32_t cDim2 = uint32_t(cDims[2]);
        uint32_t cDim3 = uint32_t(cDims[3]);
        uint32_t prod123 = uint32_t(cDims[1] * cDims[2] * cDims[3]);
        uint32_t prod23 = uint32_t(cDims[2] * cDims[3]);
        uint32_t bDim0 = uint32_t(bDims[0]);
        uint32_t bDim1 = uint32_t(bDims[1]);
        uint32_t bDim2 = uint32_t(bDims[2]);
        uint32_t bDim3 = uint32_t(bDims[3]);
        uint32_t cStride0 = uint32_t(cStrides[0]);
        uint32_t cStride1 = uint32_t(cStrides[1]);
        uint32_t cStride2 = uint32_t(cStrides[2]);
        uint32_t aStride0 = uint32_t(aStrides[0]);
        uint32_t aStride1 = uint32_t(aStrides[1]);
        uint32_t aStride2 = uint32_t(aStrides[2]);
        uint32_t aStride3 = uint32_t(aStrides[3]);
        uint32_t bStride0 = uint32_t(bStrides[0]);
        uint32_t bStride1 = uint32_t(bStrides[1]);
        uint32_t bStride2 = uint32_t(bStrides[2]);
        uint32_t bStride3 = uint32_t(bStrides[3]);

        uint32_t cDim1_fd0, cDim1_fd1;
        ocl::MakeFastDiv(int64_t(cDim1), cDim1_fd0, cDim1_fd1);
        uint32_t cDim2_fd0, cDim2_fd1;
        ocl::MakeFastDiv(int64_t(cDim2), cDim2_fd0, cDim2_fd1);
        uint32_t cDim3_fd0, cDim3_fd1;
        ocl::MakeFastDiv(int64_t(cDim3), cDim3_fd0, cDim3_fd1);
        uint32_t prod123_fd0, prod123_fd1;
        ocl::MakeFastDiv(int64_t(prod123), prod123_fd0, prod123_fd1);
        uint32_t prod23_fd0, prod23_fd1;
        ocl::MakeFastDiv(int64_t(prod23), prod23_fd0, prod23_fd1);
        uint32_t bDim0_fd0, bDim0_fd1;
        ocl::MakeFastDiv(int64_t(bDim0), bDim0_fd0, bDim0_fd1);
        uint32_t bDim1_fd0, bDim1_fd1;
        ocl::MakeFastDiv(int64_t(bDim1), bDim1_fd0, bDim1_fd1);
        uint32_t bDim2_fd0, bDim2_fd1;
        ocl::MakeFastDiv(int64_t(bDim2), bDim2_fd0, bDim2_fd1);
        uint32_t bDim3_fd0, bDim3_fd1;
        ocl::MakeFastDiv(int64_t(bDim3), bDim3_fd0, bDim3_fd1);

        m_shapeInfoArgs.AddU32("DST_D0", cDim0);
        m_shapeInfoArgs.AddS32("DST_D1", cDim1);
        m_shapeInfoArgs.AddU32("DST_D1_fd0", cDim1_fd0);
        m_shapeInfoArgs.AddU32("DST_D1_fd1", cDim1_fd1);
        m_shapeInfoArgs.AddS32("DST_D2", cDim2);
        m_shapeInfoArgs.AddU32("DST_D2_fd0", cDim2_fd0);
        m_shapeInfoArgs.AddU32("DST_D2_fd1", cDim2_fd1);
        m_shapeInfoArgs.AddS32("DST_D3", cDim3);
        m_shapeInfoArgs.AddU32("DST_D3_fd0", cDim3_fd0);
        m_shapeInfoArgs.AddU32("DST_D3_fd1", cDim3_fd1);
        m_shapeInfoArgs.AddS32("DST_D123", prod123);
        m_shapeInfoArgs.AddU32("DST_D123_fd0", prod123_fd0);
        m_shapeInfoArgs.AddU32("DST_D123_fd1", prod123_fd1);
        m_shapeInfoArgs.AddS32("DST_D23", prod23);
        m_shapeInfoArgs.AddU32("DST_D23_fd0", prod23_fd0);
        m_shapeInfoArgs.AddU32("DST_D23_fd1", prod23_fd1);
        m_shapeInfoArgs.AddU32("SRC1_D0", bDim0);
        m_shapeInfoArgs.AddU32("SRC1_D0_fd0", bDim0_fd0);
        m_shapeInfoArgs.AddU32("SRC1_D0_fd1", bDim0_fd1);
        m_shapeInfoArgs.AddU32("SRC1_D1", bDim1);
        m_shapeInfoArgs.AddU32("SRC1_D1_fd0", bDim1_fd0);
        m_shapeInfoArgs.AddU32("SRC1_D1_fd1", bDim1_fd1);
        m_shapeInfoArgs.AddU32("SRC1_D2", bDim2);
        m_shapeInfoArgs.AddU32("SRC1_D2_fd0", bDim2_fd0);
        m_shapeInfoArgs.AddU32("SRC1_D2_fd1", bDim2_fd1);
        m_shapeInfoArgs.AddU32("SRC1_D3", bDim3);
        m_shapeInfoArgs.AddU32("SRC1_D3_fd0", bDim3_fd0);
        m_shapeInfoArgs.AddU32("SRC1_D3_fd1", bDim3_fd1);
        m_shapeInfoArgs.AddS32("DST_S0", cStride0);
        m_shapeInfoArgs.AddS32("DST_S1", cStride1);
        m_shapeInfoArgs.AddS32("DST_S2", cStride2);
        m_shapeInfoArgs.AddS32("SRC0_S0", aStride0);
        m_shapeInfoArgs.AddS32("SRC0_S1", aStride1);
        m_shapeInfoArgs.AddS32("SRC0_S2", aStride2);
        m_shapeInfoArgs.AddS32("SRC0_S3", aStride3);
        m_shapeInfoArgs.AddS32("SRC1_S0", bStride0);
        m_shapeInfoArgs.AddS32("SRC1_S1", bStride1);
        m_shapeInfoArgs.AddS32("SRC1_S2", bStride2);
        m_shapeInfoArgs.AddS32("SRC1_S3", bStride3);
    }
}

bool BinaryNode::MustCollapse() {
    return (base::IsRowMajor(m_aDesc) && base::IsRowMajor(m_bDesc));
}

void BinaryNode::Collapse(dnnl::memory::dims &dims, dnnl::memory::dims &strides) {
    strides[2] *= dims[2];
    strides[1] *= dims[1];
    strides[0] *= dims[0];
    dims[3] *= dims[2];
    dims[2] = dims[1];
    dims[1] = dims[0];
    dims[0] = 1;
}

void BinaryNode::InitGrid(const dnnl::memory::dims &cDims) {
    size_t cDim0 = size_t(cDims[0]);
    size_t cDim1 = size_t(cDims[1]);
    size_t cDim2 = size_t(cDims[2]);
    size_t cDim3 = size_t(cDims[3]);

    size_t blockSize = 128;
    size_t hDim3 = std::max(cDim3 / 2, size_t(1));

    size_t lws0 = std::min(hDim3, blockSize);
    size_t lws1 = std::min(cDim2, blockSize / lws0);
    size_t lws2 = std::min(std::min(cDim0 * cDim1, blockSize / lws0 / lws1), size_t(64));

    size_t gws0 = ocl::DivUp(hDim3, lws0);
    size_t gws1 = ocl::DivUp(cDim2, lws1);
    size_t gws2 = ocl::DivUp(cDim0 * cDim1, lws2);

    m_unravel = (gws1 > 65535 || gws2 > 65535);
    
    if (m_unravel) {
        lws0 = blockSize;
        lws1 = 1;
        lws2 = 1;

        gws0 = ocl::DivUp(cDim0 * cDim1 * cDim2 * cDim3, blockSize);
        gws1 = 1;
        gws2 = 1;
    }

    m_lws[0] = lws0;
    m_lws[1] = lws1;
    m_lws[2] = lws2;

    m_gws[0] = gws0 * lws0;
    m_gws[1] = gws1 * lws1;
    m_gws[2] = gws2 * lws2;
}

void BinaryNode::InitKernel() {
    std::string sig = MakeSig();
    if (FindKernel(sig, m_kernel)) {
        return;
    }
    ocl::KernelContext kernelContext;
    std::string prolog = MakeProlog();
    const char *kernelCode = 
        m_unravel ? 
            kernels::BinarySimpleUnravelKernelCode() :
            kernels::BinarySimpleKernelCode();
    m_kernel = std::make_shared<ocl::Kernel>();
    m_kernel->Init(
        m_oclContext, 
        "binary", 
        kernelContext, 
        prolog.c_str(), 
        kernelCode);
    EnterKernel(sig, m_kernel);
}

std::string BinaryNode::MakeSig() {
    NodeSigBuilder sb;
    sb.String("*binary_simple");
    sb.Int(int(m_op));
    sb.Int(int(m_aDesc.get_data_type()));
    sb.Int(int(m_bDesc.get_data_type()));
    sb.Int(int(m_cDesc.get_data_type()));
    sb.Bool(m_unravel);
    return sb.Get();
}

std::string BinaryNode::MakeProlog() {
    std::stringstream ss;

    ocl::CommonXe::EmitGrid(ss);
    ocl::CommonXe::EmitFastDiv(ss);

    std::string binOp;
    switch (m_op) {
    case BinaryOp::Add:
        ss << kernels::BinarySimpleAddOpCode();
        binOp = "op_add";
        break;
    case BinaryOp::Sub:
        ss << kernels::BinarySimpleSubOpCode();
        binOp = "op_sub";
        break;
    case BinaryOp::Mul:
        ss << kernels::BinarySimpleMulOpCode();
        binOp = "op_mul";
        break;
    case BinaryOp::Div:
        ss << kernels::BinarySimpleDivOpCode();
        binOp = "op_div";
        break;
    default:
        assert(false);
        break;
    }

    dnnl::memory::data_type aType = m_aDesc.get_data_type();
    dnnl::memory::data_type bType = m_bDesc.get_data_type();
    dnnl::memory::data_type cType = m_cDesc.get_data_type();

    ss << "#define SRC0_T " << ocl::FormatType(aType) << "\n";
    ss << "#define SRC1_T " << ocl::FormatType(bType) << "\n";
    ss << "#define DST_T " << ocl::FormatType(cType) << "\n";
    ss << "#define BIN_OP " << binOp << "\n";
    ss << "\n";

    std::string shapeInfoArgsCode = m_shapeInfoArgs.GetCode();
    ss << "#define SHAPE_INFO_ARGS " << shapeInfoArgsCode << "\n";
    ss << "\n";

    return ss.str();
}

void BinaryNode::InitNdRange() {
    m_ndRange = ocl::NdRange(m_gws[0], m_gws[1], m_gws[2], m_lws[0], m_lws[1], m_lws[2]);
}

} // namespace

//
//    BinarySimple
//

BinarySimple::BinarySimple(Context *context):
        m_context(context) { }

BinarySimple::~BinarySimple() { }

std::unique_ptr<core::Node> BinarySimple::CreateNode(
        BinaryOp op,
        core::Node *a, 
        core::Node *b,
        core::DataType dstType,
        bool inplace) {
    std::unique_ptr<BinaryNode> node = 
        std::make_unique<BinaryNode>(m_context, op, a, b, dstType, inplace);
    if (!node->Init()) {
        return nullptr;
    }
    return node;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

