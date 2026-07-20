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
#include <vector>
#include <memory>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

namespace arhat {
namespace onednn {
namespace base {

class Device;
class NodeBase;

//
//    Quantization modes
//

enum class QuantMode {
    None,
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
//    Platform
//

class Platform: public core::Platform {
public:
    Platform();
    ~Platform();
public:
    std::string Name() override;
    int DeviceCount() override;
    core::Device *GetDevice(int deviceId) override;
private:
    void CreateDevices();
    void CreateDevicesWithKind(dnnl::engine::kind kind);
private:
    std::vector<std::unique_ptr<Device>> m_devices;
};

//
//    Device
//

class Device: public core::Device {
public:
    Device(
        Platform *platform, 
        core::DeviceKind kind,
        const std::string &name,
        const std::string &description,
        int index);
    ~Device();
public:
    core::Platform *GetPlatform() override;
    core::DeviceKind Kind() override;
    std::string Name() override;
    std::string Description() override;
    std::unique_ptr<core::Context> CreateContext() override;
public:
    int Index() {
        return m_index;
    }
private:
    Platform *m_platform;
    core::DeviceKind m_kind;
    std::string m_name;
    std::string m_description;
    int m_index;
};

//
//    MemoryPool
//

class MemoryPool {
public:
    MemoryPool() { }
    virtual ~MemoryPool() { }
public:
    virtual void Reset() = 0;
    virtual void Start() = 0;
    virtual int Alloc(size_t size) = 0;
    virtual dnnl::memory Get(int index) = 0;
};

//
//    TempMemory
//

class TempMemory {
public:
    TempMemory():
        m_pool(nullptr),
        m_index(0) { }
    TempMemory(MemoryPool *pool, int index):
        m_pool(pool),
        m_index(index) { }
    TempMemory(const TempMemory &other) = default;
    ~TempMemory() { }
public:
    TempMemory &operator=(const TempMemory &other) = default;
    dnnl::memory Get() const {
        return (m_pool != nullptr) ? m_pool->Get(m_index) : dnnl::memory();
    }
private:
    MemoryPool *m_pool;
    int m_index;
};

//
//    Context
//

class Context: public core::Context {
public:
    Context(Device *device);
    ~Context();
public:
    core::Device *GetDevice() override;
    void Wait() override;
    void Reset() override;
    // node factories
    std::unique_ptr<core::Node> CreateTensor(
        core::DataType type, const core::Dims &shape) override;
    std::unique_ptr<core::Node> CreateAdd(
        core::Node *a, 
        core::Node *b,
        core::DataType dstType,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateSub(
        core::Node *a, 
        core::Node *b,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateMul(
        core::Node *a, 
        core::Node *b,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateDiv(
        core::Node *a, 
        core::Node *b,
        bool inplace) override;
    std::unique_ptr<core::Node> CreateSqr(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSqrt(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateLog(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSum(core::Node *a) override;
    std::unique_ptr<core::Node> CreateSumRows(core::Node *a) override;
    std::unique_ptr<core::Node> CreateMean(core::Node *a) override;
    std::unique_ptr<core::Node> CreateConcat(
        core::Node *a, 
        core::Node *b,
        int dim) override;
    std::unique_ptr<core::Node> CreateMulMat(
        core::Node *a, 
        core::Node *b,
        core::Prec prec) override;
    std::unique_ptr<core::Node> CreateOutProd(core::Node *a, core::Node *b) override;
    std::unique_ptr<core::Node> CreateScale(
        core::Node *a,
        float scale,
        float bias, 
        bool inplace) override;
    std::unique_ptr<core::Node> CreateClamp(
        core::Node *a,
        float min,
        float max) override;
    std::unique_ptr<core::Node> CreateLeakyRelu(
        core::Node *a, 
        float negativeSlope, 
        bool inplace) override;
    std::unique_ptr<core::Node> CreateCast(core::Node *a, core::DataType type) override;
    std::unique_ptr<core::Node> CreateCont(core::Node *a, const core::Dims &shape) override;
    std::unique_ptr<core::Node> CreateReshape(core::Node *a, const core::Dims &shape) override;
    std::unique_ptr<core::Node> CreateView(
        core::Node *a,
        const core::Dims &shape,
        const core::Dims &stride,
        int offset) override;
    std::unique_ptr<core::Node> CreatePermute(core::Node *a, const core::Dims &axes) override;
    std::unique_ptr<core::Node> CreateTranspose(core::Node *a) override;
    std::unique_ptr<core::Node> CreateSoftMax(
        core::Node *a,
        core::Node *mask,
        core::Node *sinks,
        float scale,
        float maxBias, 
        bool inplace) override;
    std::unique_ptr<core::Node> CreateConv2d(
        core::Node *a,
        core::Node *b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1) override;
    std::unique_ptr<core::Node> CreateConv3d(
        core::Node *a,
        core::Node *b,
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
        int OC) override;
    std::unique_ptr<core::Node> CreateConv2dDw(
        core::Node *a,
        core::Node *b,
        int s0,
        int s1,
        int p0,
        int p1,
        int d0,
        int d1);
    std::unique_ptr<core::Node> CreatePool2d(
        core::Node *a,
        core::PoolOp op,
        int k0,
        int k1,
        int s0,
        int s1,
        int p0,
        int p1) override;
    std::unique_ptr<core::Node> CreateUpscale(
        core::Node *a,
        const core::Dims &shape,
        core::ScaleMode mode,
        bool alignCorners) override;
    // unary op node factories
    std::unique_ptr<core::Node> CreateAbs(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateNeg(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateTanh(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateElu(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateRelu(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateSigmoid(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateGelu(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateHardswish(core::Node *a) override;
    std::unique_ptr<core::Node> CreateHardsigmoid(core::Node *a) override;
    std::unique_ptr<core::Node> CreateExp(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateGeluErf(core::Node *a, bool inplace) override;
    std::unique_ptr<core::Node> CreateRound(core::Node *a, bool inplace) override;
public:
    dnnl::engine &Engine() {
        return m_engine;
    }
    dnnl::stream &Stream() {
        return m_stream;
    }
    NodeBase *CastNode(core::Node *node);
    void MemoryPoolStart();
    TempMemory AllocTempMemory(const dnnl::memory::desc &desc);
private:
    Device *m_device;
    dnnl::engine m_engine;
    dnnl::stream m_stream;
    std::unique_ptr<MemoryPool> m_memoryPool;
};

//
//    NodeBase
//

class NodeBase: public core::Node {
public:
    NodeBase(Context *context);
    ~NodeBase();
public:
    Context *GetContext() override;
    void Read(void *data, int offset, int size) override;
    void Write(const void *data, int offset, int size) override;
    void Fill(uint8_t value) override;
public:
    dnnl::engine &Engine() {
        return m_context->Engine();
    }
    dnnl::stream &Stream() {
        return m_context->Stream();
    }
    dnnl::memory Memory() {
        return m_memory;
    }
    dnnl::memory::desc MemoryDesc() {
        return m_memory_desc;
    }
    dnnl::memory::data_type MemoryType() {
        return m_memory_desc.get_data_type();
    }
    int MemoryRank() {
        return int(m_memory_desc.get_dims().size());
    }
    dnnl::memory::dims MemoryDims();
    int MemoryVolume();
    dnnl::memory::dims RawMemoryDims();
    int RawMemoryVolume();
    QuantMode Quant() const {
        return m_quantMode;
    }
    void SetMemory(const dnnl::memory::desc &desc);
    void SetMemory(const dnnl::memory::desc &desc, const dnnl::memory &memory);
    void SetQuant(QuantMode quantMode);
private:
    void ReadRowMajor(void *data, int offset, int size);
    void WriteRowMajor(const void *data, int offset, int size);
private:
    dnnl::memory CreateTransferMemory();
protected:
    Context *m_context;
    dnnl::memory::desc m_memory_desc;
    dnnl::memory m_memory;
    QuantMode m_quantMode;
};

//
//    Reorder
//

class Reorder {
public:
    Reorder();
    ~Reorder();
public:
    void Init(
        Context *context,
        const dnnl::memory::desc &srcDesc,
        const dnnl::memory::desc &dstDesc);
    void Compute(dnnl::memory &srcMem, dnnl::memory &dstMem);
    bool IsSet();
private:
    Context *m_context;
    dnnl::memory::desc m_srcDesc;
    dnnl::memory::desc m_dstDesc;
    dnnl::reorder m_reorder;
};

//
//    Utilities
//

dnnl::memory::data_type MapDataType(core::DataType dataType);
dnnl::memory::dims MapDims(const core::Dims &dims);
int64_t BytesToItems(dnnl::memory::data_type dataType, int64_t bytes);
int64_t ItemsToBytes(dnnl::memory::data_type dataType, int64_t items);
dnnl::memory::format_tag DefaultFormatTag(int rank);
dnnl::memory::desc PlainMemoryDesc(const dnnl::memory::desc &desc);
bool IsRowMajor(const dnnl::memory::desc &desc);
int GetBlockSize(QuantMode quantMode);
int GetQuantSize(QuantMode quantMode);

//
//    Diagnostics
//

void DiagPrintMemoryDims(const char *tag, const dnnl::memory::dims &dims);
void DiagPrintMemoryDesc(const char *tag, const dnnl::memory::desc &desc);
void DiagPrintMemoryStat(const char *tag, NodeBase *node);

//
//    Setup interface
//

class ContextFactory {
public:
    ContextFactory() { }
    virtual ~ContextFactory() { }
public:
    virtual std::unique_ptr<core::Context> CreateContext(Device *device) = 0;
};

void EnterContextFactory(core::DeviceKind deviceKind, ContextFactory *contextFactory);

} // namespace base
} // namespace onednn
} // namespace arhat

