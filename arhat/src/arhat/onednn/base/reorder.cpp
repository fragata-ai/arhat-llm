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

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"

namespace arhat {
namespace onednn {
namespace base {

//
//    Reorder
//

Reorder::Reorder():
        m_context(nullptr) { }

Reorder::~Reorder() { }

void Reorder::Init(
        Context *context,
        const dnnl::memory::desc &srcDesc,
        const dnnl::memory::desc &dstDesc) {
    m_context = context;
    m_srcDesc = srcDesc;
    m_dstDesc = dstDesc;
    dnnl::engine &engine = m_context->Engine();
    dnnl::reorder::primitive_desc prim(engine, srcDesc, engine, dstDesc);
    m_reorder = dnnl::reorder(prim);
}

void Reorder::Compute(dnnl::memory &srcMem, dnnl::memory &dstMem) {
    m_reorder.execute(m_context->Stream(), srcMem, dstMem);
}

bool Reorder::IsSet() {
    return !m_srcDesc.is_zero();
}

} // namespace base
} // namespace onednn
} // namespace arhat

