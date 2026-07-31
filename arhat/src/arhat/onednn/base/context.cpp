/* 
* MIT License
*
* Copyright (c) 2020-2026 FRAGATA COMPUTER SYSTEMS AG
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

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <memory>

#include <CL/opencl.h>

#include "dnnl.hpp"
#include "dnnl_ocl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"
#include "arhat/onednn/base/quant.hpp"

constexpr bool ENABLE_LOG_MEMORY_POOL = false;

namespace arhat {
namespace onednn {
namespace base {

namespace {

//
//    MemoryPoolImpl
//

class MemoryPoolImpl: public MemoryPool {
public:
    MemoryPoolImpl(Context *context);
    ~MemoryPoolImpl();
public:
    void Reset() override;
    void Start() override;
    int Alloc(size_t size) override;
    dnnl::memory Get(int index) override;
private:
    struct Entry {
        dnnl::memory mem;
        size_t size = 0;
        bool used = false;
    };
private:
    Context *m_context;
    std::vector<Entry> m_pool;
    size_t m_size;
};

MemoryPoolImpl::MemoryPoolImpl(Context *context):
        m_context(context), m_size(0) { }

MemoryPoolImpl::~MemoryPoolImpl() { }

void MemoryPoolImpl::Reset() {
    if (ENABLE_LOG_MEMORY_POOL) {
        printf("[Arhat] MemoryPool: Reset: %zd buffers: total %zd bytes\n", 
            m_pool.size(), m_size);
    }
    m_pool.clear();
    m_size = 0;
}

void MemoryPoolImpl::Start() {
    for (Entry &entry: m_pool) {
        entry.used = false;
    }
}

int MemoryPoolImpl::Alloc(size_t size) {
    size_t bestDiff = std::numeric_limits<size_t>::max();
    int found = -1;
    int count = int(m_pool.size());
    for (int i = 0; i < count; i++) {
        Entry &entry = m_pool[i];
        if (!entry.used && entry.size >= size) {
            size_t diff = entry.size - size;
            if (diff < bestDiff) {
                bestDiff = diff;
                found = i;
                if (bestDiff == 0) {
                    break;
                }
            }
        }
    }
    if (found >= 0) {
        if (ENABLE_LOG_MEMORY_POOL) {
            printf("[Arhat] MemoryPool: Alloc: size %zd -> found %d\n", size, found);
        }
        m_pool[found].used = true;
        return found;
    } 
    size_t adjSize = size_t(1.05 * size);
    adjSize = ((adjSize + 255) / 256) * 256;
    dnnl::memory::desc desc(
        {dnnl::memory::dim(adjSize)}, 
        dnnl::memory::data_type::u8, 
        dnnl::memory::format_tag::a);
    dnnl::engine &engine = m_context->Engine();
    dnnl::memory mem(desc, engine);
    m_pool.push_back({mem, adjSize, true});
    m_size += adjSize;
    if (ENABLE_LOG_MEMORY_POOL) {
        printf("[Arhat] MemoryPool: Alloc: size %zd -> index %d adjSize %zd total size %zd\n", 
            size, count, adjSize, m_size); 
    }
    return count;
}

dnnl::memory MemoryPoolImpl::Get(int index) {
    return m_pool[index].mem;
}

} // namespace

//
//    Context
//

Context::Context(Device *device):
        m_device(device),
        m_bufferIndex(0) {
    core::DeviceKind deviceKind = device->Kind();
    dnnl::engine::kind kind =
        (deviceKind == core::DeviceKind::Cpu) ?
            dnnl::engine::kind::cpu :
            dnnl::engine::kind::gpu;
    int index = device->Index();
    m_engine = dnnl::engine(kind, index);
    m_stream = dnnl::stream(m_engine);
    CreateMemoryPool();
    CreateBufferManager();
}

Context::~Context() { }

core::Device *Context::GetDevice() {
    return m_device;
}

void Context::Wait() {
    m_stream.wait();
}

void Context::Reset() {
    core::Context::Reset();
    m_memoryPool->Reset();
    // must not reset m_bufferManager
}

NodeBase *Context::CastNode(core::Node *node) {
    if (node == nullptr) {
        return nullptr;
    }
    if (node->GetContext() != this) {
        core::Error("Node context mismatch");
    }
    return static_cast<NodeBase *>(node);
}

void Context::MemoryPoolStart() {
    m_memoryPool->Start();
}

TempMemory Context::AllocTempMemory(const dnnl::memory::desc &desc) {
    dnnl::memory::dims dims = desc.get_padded_dims();
    size_t nelems = 1;
    for (dnnl::memory::dim d: dims) {
        nelems *= d;
    }
    size_t bytes = desc.get_size();
    int64_t items = BytesToItems(desc.get_data_type(), int64_t(bytes));
    if (nelems != items) {
        core::Error("Temporary memory must have dense layout");
    }
    int index = m_memoryPool->Alloc(bytes);
    return TempMemory(m_memoryPool.get(), index);
}

dnnl::memory Context::GetMemory(const dnnl::memory::desc &desc) {
    return m_bufferManager->GetMemory(m_bufferIndex, desc);
}

void Context::CreateMemoryPool() {
    m_memoryPool = std::make_unique<MemoryPoolImpl>(this);
}

//
//    NodeBase
//

NodeBase::NodeBase(Context *context):
        m_context(context),
        m_quantMode(QuantMode::None) { }

NodeBase::~NodeBase() { }

Context *NodeBase::GetContext() {
    return m_context;
}

void NodeBase::Read(void *data, int offset, int size) {
    if (IsRowMajor(MemoryDesc())) {
        ReadRowMajor(data, offset, size);
        return;
    }
    if (offset != 0) {
        core::NotImplemented("Read with non-zero offset");
    }
    int64_t sizeItems = BytesToItems(MemoryType(), size);
    if (sizeItems > RawMemoryVolume()) {
        core::Error("Read is out of range");
    }
    dnnl::engine &engine = m_context->Engine();
    dnnl::stream &stream = m_context->Stream();
    dnnl::memory dstMem = CreateTransferMemory();
    dnnl::reorder::primitive_desc prim(
        engine, 
        m_memory_desc,
        dstMem.get_engine(), 
        dstMem.get_desc()); 
    dnnl::reorder reorder(prim);
    reorder.execute(stream, m_memory, dstMem);
    stream.wait();
    memcpy(data, dstMem.get_data_handle(), size);
}

void NodeBase::Write(const void *data, int offset, int size) {
    if (IsRowMajor(MemoryDesc())) {
        WriteRowMajor(data, offset, size);
        return;
    }
    if (offset != 0) {
        core::NotImplemented("Write with non-zero offset");
    }
    int64_t sizeItems = BytesToItems(MemoryType(), size);
    if (sizeItems > RawMemoryVolume()) {
        core::Error("Write is out of range");
    }
    dnnl::engine &engine = m_context->Engine();
    dnnl::stream &stream = m_context->Stream();
    dnnl::memory srcMem = CreateTransferMemory();
    memcpy(srcMem.get_data_handle(), data, size);
    dnnl::reorder::primitive_desc prim(
        srcMem.get_engine(), 
        srcMem.get_desc(), 
        engine, 
        m_memory_desc);
    dnnl::reorder reorder(prim);
    reorder.execute(stream, srcMem, m_memory);
    stream.wait();
}

void NodeBase::Fill(uint8_t value) {
    dnnl::engine &engine = m_context->Engine();
    dnnl::stream &stream = m_context->Stream();
    dnnl::memory srcMem = CreateTransferMemory();
    memset(srcMem.get_data_handle(), value, srcMem.get_desc().get_size());
    dnnl::reorder::primitive_desc prim(
        srcMem.get_engine(), 
        srcMem.get_desc(), 
        engine, 
        m_memory_desc);
    dnnl::reorder reorder(prim);
    reorder.execute(stream, srcMem, m_memory);
    stream.wait();
}

dnnl::memory::dims NodeBase::MemoryDims() {
    dnnl::memory::dims dims = m_memory_desc.get_dims();
    assert(dims.size() == 4);
    if (m_quantMode != QuantMode::None) {
        int blockSize = GetBlockSize(m_quantMode);
        int quantSize = GetQuantSize(m_quantMode);
        assert(dims[3] % blockSize == 0);
        dims[3] = (dims[3] / blockSize) * quantSize;
    }
    return dims;
}

int NodeBase::MemoryVolume() {
    int volume = 1;
    for (auto dim: MemoryDims()) {
        volume *= int(dim);
    }
    return volume;
}

dnnl::memory::dims NodeBase::RawMemoryDims() {
    return m_memory_desc.get_dims();
}

int NodeBase::RawMemoryVolume() {
    int volume = 1;
    for (auto dim: RawMemoryDims()) {
        volume *= int(dim);
    }
    return volume;
}

void NodeBase::SetMemory(const dnnl::memory::desc &desc) {
    m_memory_desc = desc;
    m_memory = m_context->GetMemory(desc);
}

void NodeBase::SetMemory(const dnnl::memory::desc &desc, const dnnl::memory &memory) {
    m_memory_desc = desc;
    m_memory = memory;
}

void NodeBase::SetQuant(QuantMode quantMode) {
    m_quantMode = quantMode;
}

dnnl::memory NodeBase::CreateTransferMemory() {
    dnnl::engine &engine = m_context->Engine();
    dnnl::engine dstEng;
    if (engine.get_kind() == dnnl::engine::kind::cpu) {
        dstEng = engine;
    } else {
        dstEng = dnnl::engine(dnnl::engine::kind::cpu, 0);
    }
    dnnl::memory::data_type dataType = MemoryType();
    dnnl::memory::dims dims = RawMemoryDims();
    int rank = MemoryRank();
    dnnl::memory::format_tag dstTag = DefaultFormatTag(rank);
    dnnl::memory::desc dstDesc(dims, dataType, dstTag);
    return dnnl::memory(dstDesc, dstEng);
}

void NodeBase::ReadRowMajor(void *data, int offset, int size) {
    dnnl::engine &engine = m_context->Engine();
    dnnl::memory::data_type dt = MemoryType();
    int64_t sizeItems = BytesToItems(dt, size);
    int64_t offsetItems = BytesToItems(dt, offset);
    int volume = RawMemoryVolume();
    if (offsetItems + sizeItems > volume) {
        core::Error("Read is out of range");
    }
    if (engine.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(m_memory.get_data_handle());
        if (src == nullptr) {
            core::Error("Null memory data handle");
        }
        memcpy(data, src + offset, size);
    } else if (engine.get_kind() == dnnl::engine::kind::gpu) {
        cl_command_queue queue = dnnl::ocl_interop::get_command_queue(m_context->Stream());
        cl_mem src = dnnl::ocl_interop::get_mem_object(m_memory);
        if (src == nullptr) {
            core::Error("Null memory data handle");
        }
        cl_int status =
            clEnqueueReadBuffer(
                queue,
                src,
                CL_TRUE,
                offset,
                size,
                data,
                0,
                nullptr,
                nullptr);
        if (status != CL_SUCCESS) {
            core::Error("OpenCL error: status %d", int(status));
        }        
    } else {
        assert(false);
    }
}

void NodeBase::WriteRowMajor(const void *data, int offset, int size) {
    dnnl::engine &engine = m_context->Engine();
    dnnl::memory::data_type dt = MemoryType();
    int64_t sizeItems = BytesToItems(dt, size);
    int64_t offsetItems = BytesToItems(dt, offset);
    int volume = RawMemoryVolume();
    if (offsetItems + sizeItems > volume) {
        core::Error("Write is out of range");
    }
    if (engine.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(m_memory.get_data_handle());
        if (dst == nullptr) {
            core::Error("Null memory data handle");
        }
        memcpy(dst + offset, data, size);
    } else if (engine.get_kind() == dnnl::engine::kind::gpu) {
        cl_command_queue queue = dnnl::ocl_interop::get_command_queue(m_context->Stream());
        cl_mem dst = dnnl::ocl_interop::get_mem_object(m_memory);
        if (dst == nullptr) {
            core::Error("Null memory data handle");
        }
        cl_int status =
            clEnqueueWriteBuffer(
                queue,
                dst,
                CL_TRUE,
                offset,
                size,
                data,
                0,
                nullptr,
                nullptr);
        if (status != CL_SUCCESS) {
            core::Error("OpenCL error: status %d", int(status));
        }        
    } else {
        assert(false);
    }
}

} // namespace base
} // namespace onednn
} // namespace arhat

