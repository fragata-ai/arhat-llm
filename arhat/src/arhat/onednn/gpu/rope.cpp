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

#include <array>
#include <memory>

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/rope.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateRope(
        core::Node *a,
        core::Node *b,
        core::Node *c,
        int nDims,
        core::RopeMode mode,
        int nCtxOrig,
        float freqBase,
        float freqScale,
        float extFactor,
        float attnFactor,
        float betaFast,
        float betaSlow,
        const std::array<int, core::MropeSections> &sections, 
        bool inplace) {
    std::unique_ptr<core::Node> node;
    RopeSimple ropeSimple(this);
    node = 
        ropeSimple.CreateNode(
            a,
            b,
            c,
            nDims,
            mode,
            nCtxOrig,
            freqBase,
            freqScale,
            extFactor,
            attnFactor,
            betaFast,
            betaSlow,
            sections, 
            inplace);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: ROPE");
    return nullptr;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

