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

#include <memory>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/gpu/runtime.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

class MulMatIdHelper {
public:
    MulMatIdHelper() { }
    virtual ~MulMatIdHelper() { }
public:
    virtual void Compute(
        const dnnl::memory &ids,
        const dnnl::memory &idsA,
        const dnnl::memory &idsC,
        const dnnl::memory &expertBounds) = 0;
};

std::unique_ptr<MulMatIdHelper> CreateMulMatIdHelper(
    Context *context,
    const dnnl::memory::desc &aDesc,
    const dnnl::memory::desc &bDesc,
    const dnnl::memory::desc &idsDesc,
    const dnnl::memory::desc &idsADesc,
    const dnnl::memory::desc &idsCDesc,
    const dnnl::memory::desc &expertBoundsDesc);

} // namespace gpu
} // namespace onednn
} // namespace arhat

