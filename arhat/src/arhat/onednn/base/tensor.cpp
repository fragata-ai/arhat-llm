/* 
* MIT License
*
* Copyright (c) 2020-2026 FRAGATA COMPUTER SYSTEMS AG
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

#include <cassert>
#include <memory>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"
#include "arhat/onednn/base/quant.hpp"

namespace arhat {
namespace onednn {
namespace base {

namespace {

//
//    Utiity functions
//

QuantMode GetQuantMode(core::DataType type) {
    switch (type) {
    case core::DataType::Q2_K:
        return QuantMode::Q2_K;
    case core::DataType::Q3_K:
        return QuantMode::Q3_K;
    case core::DataType::Q4_0:
        return QuantMode::Q4_0;
    case core::DataType::Q4_1:
        return QuantMode::Q4_1;
    case core::DataType::Q4_K:
        return QuantMode::Q4_K;
    case core::DataType::Q5_0:
        return QuantMode::Q5_0;
    case core::DataType::Q5_1:
        return QuantMode::Q5_1;
    case core::DataType::Q5_K:
        return QuantMode::Q5_K;
    case core::DataType::Q6_K:
        return QuantMode::Q6_K;
    case core::DataType::Q8_0:
        return QuantMode::Q8_0;
    case core::DataType::Q8_1:
        return QuantMode::Q8_1;
    case core::DataType::MXFP4:
        return QuantMode::MXFP4;
    default:
        return QuantMode::None;
    }
}

dnnl::memory::dims MapQuantDims(QuantMode quantMode, const core::Dims &dims) {
    int blockSize = GetBlockSize(quantMode);
    int quantSize = GetQuantSize(quantMode);
    assert(dims[0] % quantSize == 0);
    int dim0 = (dims[0] / quantSize) * blockSize;
    return dnnl::memory::dims {
        dnnl::memory::dim(dims[3]),
        dnnl::memory::dim(dims[2]),
        dnnl::memory::dim(dims[1]),
        dnnl::memory::dim(dim0)
    };    
}

//
//    TensorNode
//

class TensorNode: public NodeBase {
public:
    TensorNode(
        Context *context,
        core::DataType type, 
        const core::Dims &shape);
    ~TensorNode();
public:
    void Compute() override;
public:
    void Init();
private:
    core::DataType m_type;
    core::Dims m_shape;
};

TensorNode::TensorNode(
        Context *context,
        core::DataType type, 
        const core::Dims &shape):
            NodeBase(context),
            m_type(type),
            m_shape(shape) { }

TensorNode::~TensorNode() { }

void TensorNode::Init() {
    QuantMode quantMode = GetQuantMode(m_type);
    dnnl::memory::dims dims = 
        (quantMode == QuantMode::None) ?
            MapDims(m_shape) :
            MapQuantDims(quantMode, m_shape);
    dnnl::memory::data_type type = MapDataType(m_type);
    dnnl::memory::format_tag tag = dnnl::memory::format_tag::abcd;
    dnnl::memory::desc desc(dims, type, tag);
    SetMemory(desc);
    SetQuant(quantMode);
}

void TensorNode::Compute() {
    // nothing to do
}

} // namespace

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateTensor(
        core::DataType type, const core::Dims &shape) {
    std::unique_ptr<TensorNode> node = std::make_unique<TensorNode>(this, type, shape);
    node->Init();
    return node;
}

} // namespace base
} // namespace onednn
} // namespace arhat

