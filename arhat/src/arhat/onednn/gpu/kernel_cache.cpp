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

#include <cassert>
#include <string>
#include <memory>

#include "arhat/onednn/ocl/kernel.hpp"

#include "arhat/onednn/gpu/kernel_cache.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    KernelCache
//

KernelCache::KernelCache() { }

KernelCache::~KernelCache() { }

void KernelCache::Enter(
        const std::string &sig,
        const std::shared_ptr<ocl::Kernel> &kernel) {
    auto ret = m_cache.emplace(sig, Entry{kernel, {}});
    assert(ret.second);
}

void KernelCache::Enter(
        const std::string &sig,
        const std::shared_ptr<ocl::Kernel> &kernel,
        const ocl::NdRange &ndRange) {
    auto ret = m_cache.emplace(sig, Entry{kernel, ndRange});
    assert(ret.second);
}

bool KernelCache::Find(
        const std::string &sig,
        std::shared_ptr<ocl::Kernel> &kernel) {
    auto it = m_cache.find(sig);
    if (it == m_cache.end()) {
        return false;
    }
    kernel = it->second.kernel;
    return true;
}

bool KernelCache::Find(
        const std::string &sig,
        std::shared_ptr<ocl::Kernel> &kernel,
        ocl::NdRange &ndRange) {
    auto it = m_cache.find(sig);
    if (it == m_cache.end()) {
        return false;
    }
    kernel = it->second.kernel;
    ndRange = it->second.ndRange;
    return true;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

