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
#include "arhat/onednn/gpu/unary.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

//
//    Context
//

std::unique_ptr<core::Node> Context::CreateSqr(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Sqr;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateSqr(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateSqrt(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Sqrt;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateSqrt(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateLog(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Log;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateLog(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateSin(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Sin;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: SIN");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateCos(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Cos;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: COS");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateAbs(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Abs;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateAbs(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateSgn(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Sgn;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: SGN");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateNeg(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Neg;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateNeg(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateStep(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Step;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: STEP");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateTanh(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Tanh;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateTanh(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateElu(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Elu;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateElu(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateRelu(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Relu;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateRelu(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateSigmoid(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Sigmoid;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateSigmoid(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateGelu(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Gelu;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateGelu(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateGeluQuick(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::GeluQuick;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: GELU_QUICK");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateSilu(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Silu;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: SILU");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateHardswish(core::Node *a) {
    UnaryOp op = UnaryOp::Hardswish;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, false, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateHardswish(a);
}

std::unique_ptr<core::Node> Context::CreateHardsigmoid(core::Node *a) {
    UnaryOp op = UnaryOp::Hardsigmoid;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, false, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateHardsigmoid(a);
}

std::unique_ptr<core::Node> Context::CreateExp(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Exp;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateExp(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateExpm1(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Expm1;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: EXPM1");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateSoftplus(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Softplus;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: SOFTPLUS");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateGeluErf(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::GeluErf;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    return base::Context::CreateGeluErf(a, inplace);
}

std::unique_ptr<core::Node> Context::CreateXielu(
        core::Node *a,
        float alphaN,
        float alphaP,
        float beta,
        float eps) {
    UnaryOp op = UnaryOp::Xielu;
    UnaryParam param(alphaN, alphaP, beta, eps);
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, false, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: XIELU");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateFloor(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Floor;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: FLOOR");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateCeil(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Ceil;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: CEIL");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateRound(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Round;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: ROUND");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateTrunc(core::Node *a, bool inplace) {
    UnaryOp op = UnaryOp::Trunc;
    UnaryParam param;
    std::unique_ptr<core::Node> node;
    UnarySimple unarySimple(this);
    node = unarySimple.CreateNode(op, a, inplace, param);
    if (node != nullptr) {
        return node;
    }
    core::Error("Unsupported op: TRUNC");
    return nullptr;
}

} // namespace gpu
} // namespace onednn
} // namespace arhat

