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

#include "dnnl.hpp"

#include "arhat/onednn/base/runtime.hpp"

#include "arhat/onednn/gpu/memory_desc.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    MemoryDescUtil
//

size_t MemoryDescUtil::Nelems(const dnnl::memory::desc &md, bool withPadding) {
    dnnl::memory::dims dims = withPadding ? md.get_padded_dims() : md.get_dims();
    size_t nelems = 1;
    for (dnnl::memory::dim d: dims) {
        nelems *= d;
    }
    return nelems;
}

bool MemoryDescUtil::IsBlocked(const dnnl::memory::desc &md) {
    return (md.get_format_kind() == dnnl::memory::format_kind::blocked);
}

bool MemoryDescUtil::IsPlain(const dnnl::memory::desc &md) {
    return (IsBlocked(md) && md.get_inner_nblks() == 0);
}

bool MemoryDescUtil::IsDense(const dnnl::memory::desc &md) {
    size_t nelems = Nelems(md, true);
    size_t bytes = md.get_size();
    int64_t items = base::BytesToItems(md.get_data_type(), int64_t(bytes));
    return (nelems == items);
}

bool MemoryDescUtil::HasDenseRows(const dnnl::memory::desc &md) {
    dnnl::memory::dims strides = md.get_strides();
    size_t rank = strides.size();
    return (rank > 0 && strides[rank - 1] == 1);
}

bool MemoryDescUtil::SameDims(const dnnl::memory::desc &md1, const dnnl::memory::desc &md2) {
    dnnl::memory::dims dims1 = md1.get_dims();
    dnnl::memory::dims dims2 = md2.get_dims();
    size_t ndims = dims1.size();
    if (ndims != dims2.size()) {
        return false;
    }
    for (size_t i = 0; i < ndims; i++) {
        if (dims1[i] != dims2[i]) {
            return false;
        }
    }
    return true;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

