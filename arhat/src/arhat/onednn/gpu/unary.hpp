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

#include <array>
#include <memory>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/ocl/kernel.hpp"

#include "arhat/onednn/gpu/runtime.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

enum class UnaryOp {
    Sqr,
    Sqrt,
    Log,
    Sin,
    Cos,
    Abs,
    Sgn,
    Neg,
    Step,
    Tanh,
    Elu,
    Relu,
    Sigmoid,
    Gelu,
    GeluQuick,
    Silu,
    Hardswish,
    Hardsigmoid,
    Exp,
    Expm1,
    Softplus,
    GeluErf,
    Xielu,
    Floor,
    Ceil,
    Round,
    Trunc
};

struct UnaryParam {
    UnaryParam():
        count(0),
        param{0.0f, 0.0f} { }
    UnaryParam(float p0, float p1, float p2, float p3):
        count(4),
        param{p0, p1, p2, p3} { }
    int count;
    std::array<float, 4> param;
};

class UnarySimple {
public:
    UnarySimple(Context *context);
    ~UnarySimple();
public:
    std::unique_ptr<core::Node> CreateNode(
        UnaryOp op,
        core::Node *a, 
        bool inplace,
        const UnaryParam &param);
private:
    Context *m_context;
};

} // namespace gpu
} // namespace onednn
} // namespace arhat

