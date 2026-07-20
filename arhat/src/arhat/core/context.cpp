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

namespace arhat {
namespace core {

//
//    Context
//

Context::Context() { }

Context::~Context() { }

void Context::Reset() {
    // nothing to do: subclasses may override
}

// node factories

std::unique_ptr<Node> Context::CreateTensor(DataType type, const Dims &shape) {
    NotImplemented("Tensor");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateDup(Node *a, bool inplace) {
    NotImplemented("Dup");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateAdd(
        Node *a, 
        Node *b,
        DataType dstType,
        bool inplace) {
    NotImplemented("Add");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateAddId(
        Node *a, 
        Node *b, 
        Node *ids) {
    NotImplemented("AdddId");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateAdd1(
        Node *a, 
        Node *b,
        bool inplace) {
    NotImplemented("Add1");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateAcc(
        Node *a,
        Node *b,
        const Dims &stride,
        int offset,
        bool inplace) {
    NotImplemented("Acc");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSub(
        Node *a, 
        Node *b,
        bool inplace) {
    NotImplemented("Sub");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateMul(
        Node *a, 
        Node *b,
        bool inplace) {
    NotImplemented("Mul");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateDiv(
        Node *a, 
        Node *b,
        bool inplace) {
    NotImplemented("Div");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSqr(Node *a, bool inplace) {
    NotImplemented("Sqr");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSqrt(Node *a, bool inplace) {
    NotImplemented("Sqrt");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateLog(Node *a, bool inplace) {
    NotImplemented("Log");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSin(Node *a, bool inplace) {
    NotImplemented("Sin");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCos(Node *a, bool inplace) {
    NotImplemented("Cos");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSum(Node *a) {
    NotImplemented("Sum");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSumRows(Node *a) {
    NotImplemented("SumRows");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCumSum(Node *a) {
    NotImplemented("CumSum");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateMean(Node *a) {
    NotImplemented("Mean");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateArgmax(Node *a) {
    NotImplemented("Argmax");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCountEqual(Node *a, Node *b) {
    NotImplemented("CountEqual");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRepeat(Node *a, const Dims &shape) {
    NotImplemented("Repeat");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRepeatBack(Node *a, const Dims &shape) {
    NotImplemented("RepeatBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateConcat(
        Node *a, 
        Node *b, 
        int dim) {
    NotImplemented("Concat");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSiluBack(Node *a, Node *b) {
    NotImplemented("SiluBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateNorm(
        Node *a, 
        float eps, 
        bool inplace) {
    NotImplemented("Norm");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRmsNorm(
        Node *a, 
        float eps, 
        bool inplace) {
    NotImplemented("RmsNorm");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRmsNormBack(
        Node *a,
        Node *b,
        float eps) {
    NotImplemented("RmsNormBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGroupNorm(
        Node *a, 
        int nGroups,
        float eps, 
        bool inplace) {
    NotImplemented("GroupNorm");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateL2Norm(
        Node *a, 
        float eps, 
        bool inplace) {
    NotImplemented("L2Norm");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateMulMat(
        Node *a, 
        Node *b,
        Prec prec) {
    NotImplemented("MulMat");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateMulMatId(
        Node *a,
        Node *b,
        Node *ids) {
    NotImplemented("MulMatId");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateOutProd(Node *a, Node *b) {
    NotImplemented("OutProd");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateScale(
        Node *a,
        float scale,
        float bias, 
        bool inplace) {
    NotImplemented("Scale");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSet(
        Node *a,
        Node *b,
        const Dims &stride,
        int offset,
        bool inplace) {
    NotImplemented("Set");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCpy(Node *a, Node *b) {
    NotImplemented("Cpy");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCast(Node *a, DataType type) {
    NotImplemented("Cast");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCont(Node *a, const Dims &shape) {
    NotImplemented("Cont");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateReshape(Node *a, const Dims &shape) {
    NotImplemented("Reshape");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateView(
        Node *a,
        const Dims &shape,
        const Dims &stride,
        int offset) {
    NotImplemented("View");
    return nullptr;
}

std::unique_ptr<Node> Context::CreatePermute(Node *a, const Dims &axes) {
    NotImplemented("Permute");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateTranspose(Node *a) {
    NotImplemented("Transpose");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGetRows(Node *a, Node *b) {
    NotImplemented("GetRows");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGetRowsBack(
        Node *a, 
        Node *b,
        const Dims &shape) {
    NotImplemented("GetRowsBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSetRows(
        Node *a, 
        Node *b,
        Node *c) {
    NotImplemented("SetRows");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateDiag(Node *a) {
    NotImplemented("Diag");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateDiagMaskInf(
        Node *a,
        int nPast,
        bool inplace) {
    NotImplemented("DiagMaskInf");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateDiagMaskZero(
        Node *a,
        int nPast,
        bool inplace) {
    NotImplemented("DiagMaskZero");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSoftMax(
        Node *a,
        Node *mask,
        Node *sinks,
        float scale,
        float maxBias, 
        bool inplace) {
    NotImplemented("SoftMax");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSoftMaxBack(
        Node *a,
        Node *b,
        float scale,
        float maxBias, 
        bool inplace) {
    NotImplemented("SoftMaxBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRope(
        Node *a,
        Node *b,
        Node *c,
        int nDims,
        RopeMode mode,
        int nCtxOrig,
        float freqBase,
        float freqScale,
        float extFactor,
        float attnFactor,
        float betaFast,
        float betaSlow,
        const std::array<int, MropeSections> &sections, 
        bool inplace) {
    NotImplemented("Rope");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRopeBack(
        Node *a,
        Node *b,
        Node *c,
        int nDims,
        RopeMode mode,
        int nCtxOrig,
        float freqBase,
        float freqScale,
        float extFactor,
        float attnFactor,
        float betaFast,
        float betaSlow,
        const std::array<int, MropeSections> &sections) {
    NotImplemented("RopeBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateClamp(
        Node *a,
        float min,
        float max) {
    NotImplemented("Clamp");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateConvTranspose1d(
        Node *a,
        Node *b,
        int s0,
        int p0,
        int d0) {
    NotImplemented("Transpose1d");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateIm2col(
        Node *a,
        Node *b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1,
        bool is2d,
        DataType dstType) {
    NotImplemented("Im2col");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateIm2colBack(
        Node *a,
        Node *b,
        const Dims &shape,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1,
        bool is2d) {
    NotImplemented("Im2colBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateIm2col3d(
        Node *a,
        Node *b,
        int s0,
        int s1,
        int s2,
        int p0,
        int p1,
        int p2,
        int d0,
        int d1,
        int d2,
        int IC,
        DataType dstType) {
    NotImplemented("Im2col3d");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateConv2d(
        Node *a,    // kernel
        Node *b,    // data
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) {
    NotImplemented("Conv2d");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateConv3d(
        Node *a,    // kernel
        Node *b,    // data
        int s0,
        int s1,
        int s2,
        int p0,
        int p1,
        int p2,
        int d0,
        int d1,
        int d2,
        int C,
        int N,
        int OC) {
    NotImplemented("Conv3d");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateConv2dDw(
        Node *a,    // kernel
        Node *b,    // data
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) {
    NotImplemented("Conv2dDw");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateConvTranspose2d(
        Node *a, 
        Node *b,
        int stride) {
    NotImplemented("Transpose2d");
    return nullptr;
}

std::unique_ptr<Node> Context::CreatePool1d(
        Node *a,
        PoolOp op,
        int k0,
        int s0,
        int p0) {
    NotImplemented("Pool1d");
    return nullptr;
}

std::unique_ptr<Node> Context::CreatePool2d(
        Node *a,
        PoolOp op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1) {
    NotImplemented("Pool2d");
    return nullptr;
}

std::unique_ptr<Node> Context::CreatePool2dBack(
        Node *a,
        Node *af,    // forward tensor
        PoolOp op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1) {
    NotImplemented("Pool2dBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateUpscale(
        Node *a,
        const Dims &shape,
        ScaleMode mode,
        bool alignCorners) {
    NotImplemented("Upscale");
    return nullptr;
}

std::unique_ptr<Node> Context::CreatePad(
        Node *a,
        int lp0,
        int rp0,
        int lp1,
        int rp1,
        int lp2,
        int rp2,
        int lp3,
        int rp3,
        bool circular) {
    NotImplemented("Pad");
    return nullptr;
}

std::unique_ptr<Node> Context::CreatePadReflect1d(
        Node *a,
        int p0,
        int p1) {
    NotImplemented("Reflect1d");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRoll(Node *a, const Dims &shift) {
    NotImplemented("Roll");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateArange(
        float start,
        float stop,
        float step) {
    NotImplemented("Arange");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateTimestepEmbedding(
        Node *timesteps,
        int dim,
        int maxPeriod) {
    NotImplemented("TimestepEmbedding");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateArgsort(Node *a, SortOrder order) {
    NotImplemented("Argsort");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateLeakyRelu(
        Node *a, 
        float negativeSlope, 
        bool inplace) {
    NotImplemented("LeakyRelu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateTri(Node *a, int mode) {
    NotImplemented("Tri");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateFill(
        Node *a, 
        float value,
        bool inplace) {
    NotImplemented("Fill");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateFlashAttnExt(
        Node *q,
        Node *k,
        Node *v,
        Node *mask,
        Node *sinks,
        float scale,
        float maxBias,
        float logitSoftcap,
        Prec prec) {
    NotImplemented("FlashAttnExt");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateFlashAttnBack(
        Node *q,
        Node *k,
        Node *v,
        Node *d,
        bool masked) {
    NotImplemented("FlashAttnBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSsmConv(Node *sx, Node *c) {
    NotImplemented("SsmConv");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSsmScan(
        Node *s,
        Node *x,
        Node *dt,
        Node *A,
        Node *B,
        Node *C,
        Node *ids) {
    NotImplemented("SsmScan");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateWinPart(
        Node *a,
        int np0,
        int np1,
        int w) {
    NotImplemented("WinPart");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateWinUnpart(
        Node *a, 
        int w0,
        int h0,
        int w) {
    NotImplemented("WinUnpart");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGetRelPos(
        Node *a,
        int qh,
        int kh) {
    NotImplemented("GetRelPos");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateAddRelPos(
        Node *a,
        Node *pw,
        Node *ph,
        bool inplace) {
    NotImplemented("AddRelPos");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRwkvWkv6(
        Node *k,
        Node *v,
        Node *r,
        Node *tf,
        Node *td,
        Node *state) {
    NotImplemented("RwkvWkv6");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGatedLinearAttn(
        Node *k,
        Node *v,
        Node *q,
        Node *g,
        Node *state,
        float scale) {
    NotImplemented("GatedLinearAttn");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRwkvWkv7(
        Node *r,
        Node *w,
        Node *k,
        Node *v,
        Node *a,
        Node *b,
        Node *state) {
    NotImplemented("RwkvWkv7");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSolveTri(
        Node *a,
        Node *b,
        bool left,
        bool lower,
        bool uni) {
    NotImplemented("SolveTri");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGatedDeltaNet(
        Node *q,
        Node *k,
        Node *v,
        Node *g,
        Node *beta,
        Node *state) {
    NotImplemented("GatedDeltaNet");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCrossEntropyLoss(Node *a, Node *b) {
    NotImplemented("CrossEntropyLoss");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCrossEntropyLossBack(
        Node *a,
        Node *b,
        Node *c) {
    NotImplemented("CrossEntropyLossBack");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateOptStepAdamw(
        Node *a,
        Node *grad,
        Node *m,
        Node *v,
        Node *params) {
    NotImplemented("OptStepAdamw");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateOptStepSgd(
        Node *a,
        Node *grad,
        Node *params) {
    NotImplemented("OptStepSgd");
    return nullptr;
}

// unary op node factories

std::unique_ptr<Node> Context::CreateAbs(Node *a, bool inplace) {
    NotImplemented("Abs");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSgn(Node *a, bool inplace) {
    NotImplemented("Sgn");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateNeg(Node *a, bool inplace) {
    NotImplemented("Neg");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateStep(Node *a, bool inplace) {
    NotImplemented("Step");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateTanh(Node *a, bool inplace) {
    NotImplemented("Tanh");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateElu(Node *a, bool inplace) {
    NotImplemented("Elu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRelu(Node *a, bool inplace) {
    NotImplemented("Relu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSigmoid(Node *a, bool inplace) {
    NotImplemented("Sigmoid");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGelu(Node *a, bool inplace) {
    NotImplemented("Gelu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGeluQuick(Node *a, bool inplace) {
    NotImplemented("GeluQuick");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSilu(Node *a, bool inplace) {
    NotImplemented("Silu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateHardswish(Node *a) {
    NotImplemented("Hardswish");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateHardsigmoid(Node *a) {
    NotImplemented("Hardsigmoid");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateExp(Node *a, bool inplace) {
    NotImplemented("Exp");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateExpm1(core::Node *a, bool inplace) {
    NotImplemented("Expm1");
    return nullptr;
}

std::unique_ptr<core::Node> Context::CreateSoftplus(core::Node *a, bool inplace) {
    NotImplemented("Softplus");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGeluErf(Node *a, bool inplace) {
    NotImplemented("Erf");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateXielu(
        Node *a,
        float alphaN,
        float alphaP,
        float beta,
        float eps) {
    NotImplemented("Xielu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateFloor(Node *a, bool inplace) {
    NotImplemented("Floor");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateCeil(Node *a, bool inplace) {
    NotImplemented("Ceil");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateRound(Node *a, bool inplace) {
    NotImplemented("Round");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateTrunc(Node *a, bool inplace) {
    NotImplemented("Trunc");
    return nullptr;
}

// GLU op node factories

std::unique_ptr<Node> Context::CreateReglu(
        Node *a, 
        Node *b,
        bool swapped) {
    NotImplemented("Reglu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGeglu(
        Node *a, 
        Node *b,
        bool swapped) {
    NotImplemented("Geglu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSwiglu(
        Node *a, 
        Node *b,
        bool swapped) {
    NotImplemented("Swiglu");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateSwigluOai(
        Node *a, 
        Node *b,
        bool swapped,
        float alpha,
        float limit) {
    NotImplemented("SwigluOai");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGegluErf(
        Node *a, 
        Node *b,
        bool swapped) {
    NotImplemented("GegluErf");
    return nullptr;
}

std::unique_ptr<Node> Context::CreateGegluQuick(
        Node *a, 
        Node *b,
        bool swapped) {
    NotImplemented("GegluQuick");
    return nullptr;
}

} // namespace core
} // namespace arhat

