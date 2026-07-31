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

#include <cstdint>
#include <string>
#include <array>
#include <memory>
#include <exception>

namespace arhat {
namespace core {

class Platform;
class Device;
class Context;
class Node;

//
//    Error handling
//

class RuntimeError: public std::exception {
public:
    RuntimeError(const char *msg);
    RuntimeError(const std::string &msg);
    ~RuntimeError();
public:
    const char *what() const noexcept override;
private:
    std::string m_msg;
};

void Error(const char *fmt, ...);
void NotImplemented(const char *what);

//
//    Data types
//

enum class DataType {
    Undef,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    BF16,
    F16,
    F32,
    F64,
    // quantized
    Q2_K,
    Q3_K,
    Q4_0,
    Q4_1,
    Q4_K,
    Q5_0,
    Q5_1,
    Q5_K,
    Q6_K,
    Q8_0,
    Q8_1,
    MXFP4
};

//
//    Common constants
//

enum class DeviceKind {
    Cpu,
    Gpu,
    Xpu
};

//
//    Operation-related constants
//

enum class Prec {
    Default,
    F32
};

enum class PoolOp {
    Avg,
    Max
};

enum class ScaleMode {
    Nearest,
    Bilinear,
    Bicubic
};

enum class SortOrder {
    Asc,
    Desc
};

enum class RopeMode {
    Normal,
    Neox,
    Mrope,
    Vision,
    Imrope
};

constexpr int MropeSections = 4;

//
//    Common types
//

constexpr int MaxDims = 4;

using Dims = std::array<int, MaxDims>;

//
//    Platform
//

class Platform {
public:
    Platform() { }
    virtual ~Platform() { }
public:
    static int Count();
    static Platform *Get(int platformId);
public:
    virtual std::string Name() = 0;
    virtual int DeviceCount() = 0;
    virtual Device *GetDevice(int deviceId) = 0;
};

//
//    Device
//

class Device {
public:
    Device() { }
    virtual ~Device() { }
public:
    virtual Platform *GetPlatform() = 0;
    virtual DeviceKind Kind() = 0;
    virtual std::string Name() = 0;
    virtual std::string Description() = 0;
    virtual std::unique_ptr<Context> CreateContext() = 0;
};

//
//    Context
//

class Context {
public:
    Context();
    virtual ~Context();
public:
    static constexpr size_t NULL_BUFFER_ADDR = ~size_t(0);
public:
    virtual Device *GetDevice() = 0;
    virtual void Wait() = 0;
    virtual void Reset();
    // buffer management
    virtual int CreateBuffer() = 0;
    virtual void ResetBuffer(int index) = 0;
    virtual void DeleteBuffer(int index) = 0;
    virtual void SetBuffer(int index, size_t addr) = 0;
    // node factories
    virtual std::unique_ptr<Node> CreateTensor(DataType type, const Dims &shape);
    virtual std::unique_ptr<Node> CreateDup(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateAdd(
        Node *a, 
        Node *b,
        DataType dstType,
        bool inplace);
    virtual std::unique_ptr<Node> CreateAddId(
        Node *a, 
        Node *b, 
        Node *ids);
    virtual std::unique_ptr<Node> CreateAdd1(
        Node *a, 
        Node *b,
        bool inplace);
    virtual std::unique_ptr<Node> CreateAcc(
        Node *a,
        Node *b,
        const Dims &stride,
        int offset,
        bool inplace);
    virtual std::unique_ptr<Node> CreateSub(
        Node *a, 
        Node *b,
        bool inplace);
    virtual std::unique_ptr<Node> CreateMul(
        Node *a, 
        Node *b,
        bool inplace);
    virtual std::unique_ptr<Node> CreateDiv(
        Node *a, 
        Node *b,
        bool inplace);
    virtual std::unique_ptr<Node> CreateSqr(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateSqrt(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateLog(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateSin(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateCos(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateSum(Node *a);
    virtual std::unique_ptr<Node> CreateSumRows(Node *a);
    virtual std::unique_ptr<Node> CreateCumSum(Node *a);
    virtual std::unique_ptr<Node> CreateMean(Node *a);
    virtual std::unique_ptr<Node> CreateArgmax(Node *a);
    virtual std::unique_ptr<Node> CreateCountEqual(Node *a, Node *b);
    virtual std::unique_ptr<Node> CreateRepeat(Node *a, const Dims &shape);
    virtual std::unique_ptr<Node> CreateRepeatBack(Node *a, const Dims &shape);
    virtual std::unique_ptr<Node> CreateConcat(
        Node *a, 
        Node *b,
        int dim);
    virtual std::unique_ptr<Node> CreateSiluBack(Node *a, Node *b);
    virtual std::unique_ptr<Node> CreateNorm(
        Node *a, 
        float eps, 
        bool inplace);
    virtual std::unique_ptr<Node> CreateRmsNorm(
        Node *a, 
        float eps, 
        bool inplace);
    virtual std::unique_ptr<Node> CreateRmsNormBack(
        Node *a,
        Node *b,
        float eps);
    virtual std::unique_ptr<Node> CreateGroupNorm(
        Node *a, 
        int nGroups,
        float eps, 
        bool inplace);
    virtual std::unique_ptr<Node> CreateL2Norm(
        Node *a, 
        float eps, 
        bool inplace);
    virtual std::unique_ptr<Node> CreateMulMat(
        Node *a, 
        Node *b,
        Prec prec);
    virtual std::unique_ptr<Node> CreateMulMatId(
        Node *a,
        Node *b,
        Node *ids);
    virtual std::unique_ptr<Node> CreateOutProd(Node *a, Node *b);
    virtual std::unique_ptr<Node> CreateScale(
        Node *a,
        float scale,
        float bias, 
        bool inplace);
    virtual std::unique_ptr<Node> CreateSet(
        Node *a,
        Node *b,
        const Dims &stride,
        int offset,
        bool inplace);
    virtual std::unique_ptr<Node> CreateCpy(Node *a, Node *b);
    virtual std::unique_ptr<Node> CreateCast(Node *a, DataType type);
    virtual std::unique_ptr<Node> CreateCont(Node *a, const Dims &shape);
    virtual std::unique_ptr<Node> CreateReshape(Node *a, const Dims &shape);
    virtual std::unique_ptr<Node> CreateView(
        Node *a,
        const Dims &shape,
        const Dims &stride,
        int offset);
    virtual std::unique_ptr<Node> CreatePermute(Node *a, const Dims &axes);
    virtual std::unique_ptr<Node> CreateTranspose(Node *a);
    virtual std::unique_ptr<Node> CreateGetRows(Node *a, Node *b);
    virtual std::unique_ptr<Node> CreateGetRowsBack(
        Node *a, 
        Node *b,
        const Dims &shape);
    virtual std::unique_ptr<Node> CreateSetRows(
        Node *a, 
        Node *b,
        Node *c);
    virtual std::unique_ptr<Node> CreateDiag(Node *a);
    virtual std::unique_ptr<Node> CreateDiagMaskInf(
        Node *a,
        int nPast,
        bool inplace);
    virtual std::unique_ptr<Node> CreateDiagMaskZero(
        Node *a,
        int nPast,
        bool inplace);
    virtual std::unique_ptr<Node> CreateSoftMax(
        Node *a,
        Node *mask,
        Node *sinks,
        float scale,
        float maxBias, 
        bool inplace);
    virtual std::unique_ptr<Node> CreateSoftMaxBack(
        Node *a,
        Node *b,
        float scale,
        float maxBias, 
        bool inplace);
    virtual std::unique_ptr<Node> CreateRope(
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
        bool inplace);
    virtual std::unique_ptr<Node> CreateRopeBack(
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
        const std::array<int, MropeSections> &sections);
    virtual std::unique_ptr<Node> CreateClamp(
        Node *a,
        float min,
        float max);
    virtual std::unique_ptr<Node> CreateConvTranspose1d(
        Node *a,
        Node *b,
        int s0,
        int p0,
        int d0);
    virtual std::unique_ptr<Node> CreateIm2col(
        Node *a,
        Node *b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1,
        bool is2d,
        DataType dstType);
    virtual std::unique_ptr<Node> CreateIm2colBack(
        Node *a,
        Node *b,
        const Dims &shape,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1,
        bool is2d);
    virtual std::unique_ptr<Node> CreateIm2col3d(
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
        DataType dstType);
    virtual std::unique_ptr<Node> CreateConv2d(
        Node *a,    // kernel
        Node *b,    // data
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1);
    virtual std::unique_ptr<Node> CreateConv3d(
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
        int OC);
    virtual std::unique_ptr<Node> CreateConv2dDw(
        Node *a,    // kernel
        Node *b,    // data
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1);
    virtual std::unique_ptr<Node> CreateConvTranspose2d(
        Node *a, 
        Node *b,
        int stride);
    virtual std::unique_ptr<Node> CreatePool1d(
        Node *a,
        PoolOp op,
        int k0,
        int s0,
        int p0);
    virtual std::unique_ptr<Node> CreatePool2d(
        Node *a,
        PoolOp op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1);
    virtual std::unique_ptr<Node> CreatePool2dBack(
        Node *a,
        Node *af,    // forward tensor
        PoolOp op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1);
    virtual std::unique_ptr<Node> CreateUpscale(
        Node *a,
        const Dims &shape,
        ScaleMode mode,
        bool alignCorners);
    virtual std::unique_ptr<Node> CreatePad(
        Node *a,
        int lp0,
        int rp0,
        int lp1,
        int rp1,
        int lp2,
        int rp2,
        int lp3,
        int rp3,
        bool circular);
    virtual std::unique_ptr<Node> CreatePadReflect1d(
        Node *a,
        int p0,
        int p1);
    virtual std::unique_ptr<Node> CreateRoll(Node *a, const Dims &shift);
    virtual std::unique_ptr<Node> CreateArange(
        float start,
        float stop,
        float step);
    virtual std::unique_ptr<Node> CreateTimestepEmbedding(
        Node *timesteps,
        int dim,
        int maxPeriod);
    virtual std::unique_ptr<Node> CreateArgsort(Node *a, SortOrder order);
    virtual std::unique_ptr<Node> CreateLeakyRelu(
        Node *a, 
        float negativeSlope, 
        bool inplace);
    virtual std::unique_ptr<Node> CreateTri(Node *a, int mode);
    virtual std::unique_ptr<Node> CreateFill(
        Node *a, 
        float value,
        bool inplace);
    virtual std::unique_ptr<Node> CreateFlashAttnExt(
        Node *q,
        Node *k,
        Node *v,
        Node *mask,
        Node *sinks,
        float scale,
        float maxBias,
        float logitSoftcap,
        Prec prec);
    virtual std::unique_ptr<Node> CreateFlashAttnBack(
        Node *q,
        Node *k,
        Node *v,
        Node *d,
        bool masked);
    virtual std::unique_ptr<Node> CreateSsmConv(Node *sx, Node *c);
    virtual std::unique_ptr<Node> CreateSsmScan(
        Node *s,
        Node *x,
        Node *dt,
        Node *A,
        Node *B,
        Node *C,
        Node *ids);
    virtual std::unique_ptr<Node> CreateWinPart(
        Node *a,
        int np0,
        int np1,
        int w);
    virtual std::unique_ptr<Node> CreateWinUnpart(
        Node *a, 
        int w0,
        int h0,
        int w);
    virtual std::unique_ptr<Node> CreateGetRelPos(
        Node *a,
        int qh,
        int kh);
    virtual std::unique_ptr<Node> CreateAddRelPos(
        Node *a,
        Node *pw,
        Node *ph,
        bool inplace);
    virtual std::unique_ptr<Node> CreateRwkvWkv6(
        Node *k,
        Node *v,
        Node *r,
        Node *tf,
        Node *td,
        Node *state);
    virtual std::unique_ptr<Node> CreateGatedLinearAttn(
        Node *k,
        Node *v,
        Node *q,
        Node *g,
        Node *state,
        float scale);
    virtual std::unique_ptr<Node> CreateRwkvWkv7(
        Node *r,
        Node *w,
        Node *k,
        Node *v,
        Node *a,
        Node *b,
        Node *state);
    virtual std::unique_ptr<Node> CreateSolveTri(
        Node *a,
        Node *b,
        bool left,
        bool lower,
        bool uni);
    virtual std::unique_ptr<Node> CreateGatedDeltaNet(
        Node *q,
        Node *k,
        Node *v,
        Node *g,
        Node *beta,
        Node *state);
    virtual std::unique_ptr<Node> CreateCrossEntropyLoss(Node *a, Node *b);
    virtual std::unique_ptr<Node> CreateCrossEntropyLossBack(
        Node *a,
        Node *b,
        Node *c);
    virtual std::unique_ptr<Node> CreateOptStepAdamw(
        Node *a,
        Node *grad,
        Node *m,
        Node *v,
        Node *params);
    virtual std::unique_ptr<Node> CreateOptStepSgd(
        Node *a,
        Node *grad,
        Node *params);
    // unary op node factories
    virtual std::unique_ptr<Node> CreateAbs(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateSgn(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateNeg(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateStep(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateTanh(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateElu(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateRelu(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateSigmoid(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateGelu(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateGeluQuick(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateSilu(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateHardswish(Node *a);
    virtual std::unique_ptr<Node> CreateHardsigmoid(Node *a);
    virtual std::unique_ptr<Node> CreateExp(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateExpm1(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateSoftplus(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateGeluErf(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateXielu(
        Node *a,
        float alphaN,
        float alphaP,
        float beta,
        float eps);
    virtual std::unique_ptr<Node> CreateFloor(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateCeil(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateRound(Node *a, bool inplace);
    virtual std::unique_ptr<Node> CreateTrunc(Node *a, bool inplace);
    // GLU op node factories
    virtual std::unique_ptr<Node> CreateReglu(
        Node *a, 
        Node *b,
        bool swapped);
    virtual std::unique_ptr<Node> CreateGeglu(
        Node *a, 
        Node *b,
        bool swapped);
    virtual std::unique_ptr<Node> CreateSwiglu(
        Node *a, 
        Node *b,
        bool swapped);
    virtual std::unique_ptr<Node> CreateSwigluOai(
        Node *a, 
        Node *b,
        bool swapped,
        float alpha,
        float limit);
    virtual std::unique_ptr<Node> CreateGegluErf(
        Node *a, 
        Node *b,
        bool swapped);
    virtual std::unique_ptr<Node> CreateGegluQuick(
        Node *a, 
        Node *b,
        bool swapped);
};

//
//    Node
//

class Node {
public:
    Node() { }
    virtual ~Node() { }
public:
    virtual Context *GetContext() = 0;
    virtual void Read(void *data, int offset, int size) = 0;
    virtual void Write(const void *data, int offset, int size) = 0;
    virtual void Fill(uint8_t value) = 0;
    virtual void Compute() = 0;
};

//
//    Public functions
//

bool CanReshape(
    const int64_t *srcNe,
    const size_t *srcNb,
    const int64_t *dstNe,
    const size_t *dstNb);

//
//    Setup interface
//

void EnterPlatform(Platform *platform);

} // namespace core
} // namespace arhat

