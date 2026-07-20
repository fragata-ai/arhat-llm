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
#include "arhat/onednn/gpu/mul_mat_id.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateMulMatId(
        core::Node *a, 
        core::Node *b, 
        core::Node *ids) {
    std::unique_ptr<core::Node> node;
    MulMatIdVec mulMatIdVec(this);
    node = mulMatIdVec.CreateNode(b, a, ids);
    if (node != nullptr) {
        return node;
    }
    MulMatIdMm mulMatIdMm(this);
    node = mulMatIdMm.CreateNode(b, a, ids);
    if (node != nullptr) {
        return node;
    }
    MulMatIdQuantVecV2Opt mulMatIdQuantVecV2Opt(this);
    node = mulMatIdQuantVecV2Opt.CreateNode(b, a, ids);
    if (node != nullptr) {
        return node;
    }
    MulMatIdQuantVecV2 mulMatIdQuantVecV2(this);
    node = mulMatIdQuantVecV2.CreateNode(b, a, ids);
    if (node != nullptr) {
        return node;
    }
    MulMatIdQuantVec mulMatIdQuantVec(this);
    node = mulMatIdQuantVec.CreateNode(b, a, ids);
    if (node != nullptr) {
        return node;
    }
    MulMatIdQuantMm mulMatIdQuantMm(this);
    node = mulMatIdQuantMm.CreateNode(b, a, ids);
    if (node != nullptr) {
        return node;
    }
    MulMatIdQuantSimple mulMatIdQuantSimple(this);
    node = mulMatIdQuantSimple.CreateNode(b, a, ids);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: MUL_MAT_ID");
    return nullptr;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

