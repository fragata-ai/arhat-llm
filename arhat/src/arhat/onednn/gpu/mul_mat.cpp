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

#include "arhat/onednn/base/runtime.hpp"

#include "arhat/onednn/gpu/runtime.hpp"
#include "arhat/onednn/gpu/mul_mat.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateMulMat(
        core::Node *a, 
        core::Node *b,
        core::Prec prec) {
    // prec is not used - is this correct?
    std::unique_ptr<core::Node> node;
    MulMatQuantVecV2Opt mulMatQuantVecV2Opt(this);
    node = mulMatQuantVecV2Opt.CreateNode(b, a);
    if (node != nullptr) {
        return node;
    }
#if 0 // Reserved: MMVQ seems faster than DMMV
    MulMatVecV2 mulMatVecV2(this);
    node = mulMatVecV2.CreateNode(b, a);
    if (node != nullptr) {
        return node;
    }
#endif
    MulMatQuantVecV2 mulMatQuantVecV2(this);
    node = mulMatQuantVecV2.CreateNode(b, a);
    if (node != nullptr) {
        return node;
    }
    MulMatQuantVec mulMatQuantVec(this);
    node = mulMatQuantVec.CreateNode(b, a);
    if (node != nullptr) {
        return node;
    }
    MulMatQuantMm mulMatQuantMm(this);
    node = mulMatQuantMm.CreateNode(b, a);
    if (node != nullptr) {
        return node;
    }
    MulMatQuantSimple mulMatQuantSimple(this);
    node = mulMatQuantSimple.CreateNode(b, a);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateMulMat(a, b, prec);
}

std::unique_ptr<core::Node> Context::CreateOutProd(core::Node *a, core::Node *b) {
    return base::Context::CreateOutProd(a, b);
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

