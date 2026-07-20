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

#include "dnnl.hpp"

#include "arhat/onednn/gpu/memory_desc.hpp"
#include "arhat/onednn/gpu/mul_mat_util.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    MulMatV2Util
//

bool MulMatV2Util::HasDenseRowsCols(const dnnl::memory::desc &md) {
    if (!MemoryDescUtil::IsPlain(md)) {
        return false;
    }
    dnnl::memory::dims dims = md.get_dims();
    dnnl::memory::dims strides = md.get_strides();
    return (strides[3] == 1 &&
        strides[2] == strides[3] * dims[3] &&
        (dims[0] == 1 || strides[0] == strides[1] * dims[1]));
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

