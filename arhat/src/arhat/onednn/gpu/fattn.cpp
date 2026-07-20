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

#include <memory>

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/fattn.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateFlashAttnExt(
        core::Node *q,
        core::Node *k,
        core::Node *v,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias,
        float logitSoftcap,
        core::Prec prec) {
    std::unique_ptr<core::Node> node;
    FattnVec fattnVec(this);
    node = fattnVec.CreateNode(q, k, v, mask, sinks, scale, maxBias, logitSoftcap, prec);
    if (node != nullptr) {
        return node;
    }
    FattnTile fattnTile(this);
    node = fattnTile.CreateNode(q, k, v, mask, sinks, scale, maxBias, logitSoftcap, prec);
    if (node != nullptr) {
        return node;
    }
#if 1 // TODO: Reenable this
    FattnSimple fattnSimple(this);
    node = fattnSimple.CreateNode(q, k, v, mask, sinks, scale, maxBias, logitSoftcap, prec);
    if (node != nullptr) {
        return node;
    }
#endif
    core::Error("Unsupported op: FLASH_ATTN_EXT");
    return nullptr;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

