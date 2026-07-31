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
#include <cassert>
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>

#include "dnnl.hpp"

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/base/runtime.hpp"

constexpr bool ENABLE_LOG_BUFFER_MANAGER = false;

namespace arhat {
namespace onednn {
namespace base {

namespace {

//
//    Buffer
//

struct BufferMemoryKey {
    dnnl::memory::data_type dt;
    size_t addr;
    size_t size;

    struct Hash {
        size_t operator()(const BufferMemoryKey &key) const noexcept {
            size_t h1 = std::hash<dnnl::memory::data_type>()(key.dt);
            size_t h2 = std::hash<size_t>()(key.addr);
            size_t h3 = std::hash<size_t>()(key.size);
            return Combine(Combine(h1, h2), h3);
        }
        static size_t Combine(size_t h1, size_t h2) {
            return h1 ^ (h2 << 1);
        }
    };

    struct Equal {
        bool operator()(const BufferMemoryKey &key1, const BufferMemoryKey &key2) const noexcept {
            return (key1.dt == key2.dt && key1.addr == key2.addr && key1.size == key2.size);
        }
    };
};

class Buffer {
public:
    Buffer(const dnnl::engine &engine);
    ~Buffer();
public:
    void Init();
    void Reset();
    void Free();
    bool IsFree() const {
        return m_isFree;
    }
    size_t GetTotalSize() const {
        return m_totalSize;
    }
    void SetAddr(size_t addr);
    dnnl::memory GetMemory(const dnnl::memory::desc &desc);
private:
    static bool IsPlain(const dnnl::memory::desc &desc);
private:
    dnnl::engine m_engine;
    bool m_isFree;
    size_t m_totalSize;
    int m_hits;
    int m_misses;
    std::unordered_map<
        BufferMemoryKey, 
        dnnl::memory, 
        BufferMemoryKey::Hash, 
        BufferMemoryKey::Equal> m_memoryMap;
    size_t m_currAddr;
};

Buffer::Buffer(const dnnl::engine &engine):
        m_engine(engine), 
        m_isFree(false),
        m_totalSize(0),
        m_hits(0),
        m_misses(0),
        m_currAddr(core::Context::NULL_BUFFER_ADDR) { }

Buffer::~Buffer() { }

void Buffer::Init() {
    m_isFree = false;
}

void Buffer::Reset() {
    if (ENABLE_LOG_BUFFER_MANAGER) {
        printf("[Arhat] Buffer: Reset: total %zd bytes, %d hits, %d misses\n", 
            m_totalSize, m_hits, m_misses);
    }
    assert(!m_isFree);
    m_totalSize = 0;
    m_hits = 0;
    m_misses = 0;
    m_memoryMap.clear();
    m_currAddr = core::Context::NULL_BUFFER_ADDR;
}

void Buffer::Free() {
    if (ENABLE_LOG_BUFFER_MANAGER) {
        printf("[Arhat] Buffer: Free: total %zd bytes, %d hits, %d misses\n", 
            m_totalSize, m_hits, m_misses);
    }
    assert(!m_isFree);
    m_isFree = true;
    m_totalSize = 0;
    m_hits = 0;
    m_misses = 0;
    m_memoryMap.clear();
    m_currAddr = core::Context::NULL_BUFFER_ADDR;
}

void Buffer::SetAddr(size_t addr) {
    assert(!m_isFree);
    m_currAddr = addr;
}

dnnl::memory Buffer::GetMemory(const dnnl::memory::desc &desc) {
    if (m_currAddr == core::Context::NULL_BUFFER_ADDR || !IsPlain(desc)) {
        return dnnl::memory(desc, m_engine);
    }
    BufferMemoryKey key{
        desc.get_data_type(),
        m_currAddr,
        desc.get_size()
    };
    auto it = m_memoryMap.find(key);
    if (it != m_memoryMap.end()) {
        m_hits++;
        return it->second;
    }
    dnnl::memory memory = dnnl::memory(desc, m_engine);
    m_memoryMap[key] = memory;
    m_totalSize += key.size;
    m_misses++;
    return memory;
}

bool Buffer::IsPlain(const dnnl::memory::desc &desc) {
    return (desc.get_format_kind() == dnnl::memory::format_kind::blocked &&
        desc.get_inner_nblks() == 0);
}

//
//    BufferManagerImpl
//

class BufferManagerImpl: public BufferManager {
public:
    BufferManagerImpl(Context *context);
    ~BufferManagerImpl();
public:
    int CreateBuffer() override;
    void ResetBuffer(int index) override;
    void DeleteBuffer(int index) override;
    void SetBuffer(int index, size_t addr) override;
    dnnl::memory GetMemory(int index, const dnnl::memory::desc &desc) override;
private:
    Context *m_context;
    dnnl::engine m_engine;
    std::vector<Buffer> m_buffers;
};

BufferManagerImpl::BufferManagerImpl(Context *context):
        m_context(context),
        m_engine(context->Engine()) { }

BufferManagerImpl::~BufferManagerImpl() { }

int BufferManagerImpl::CreateBuffer() {
    int index = -1;
    for (int i = int(m_buffers.size()) - 1; i >= 0; i--) {
        if (m_buffers[i].IsFree()) {
            index = i;
            m_buffers[index].Init();
            break;
        }
    }
    if (index < 0) {
        m_buffers.emplace_back(m_engine);
        index = int(m_buffers.size()) - 1;
    }
    if (ENABLE_LOG_BUFFER_MANAGER) {
        printf("[Arhat] BufferManager: CreateBuffer %d\n", index);
    }
    return index;
}

void BufferManagerImpl::ResetBuffer(int index) {
    if (ENABLE_LOG_BUFFER_MANAGER) {
        printf("[Arhat] BufferManager: ResetBuffer %d\n", index);
    }
    assert(index >= 0 && index < m_buffers.size());
    m_buffers[index].Reset();
}

void BufferManagerImpl::DeleteBuffer(int index) {
    if (ENABLE_LOG_BUFFER_MANAGER) {
        printf("[Arhat] BufferManager: DeleteBuffer %d\n", index);
    }
    assert(index >= 0 && index < m_buffers.size());
    m_buffers[index].Free();
}

void BufferManagerImpl::SetBuffer(int index, size_t addr) {
    assert(index >= 0 && index < m_buffers.size());
    m_buffers[index].SetAddr(addr);
}

dnnl::memory BufferManagerImpl::GetMemory(int index, const dnnl::memory::desc &desc) {
    assert(index >= 0 && index < m_buffers.size());
    return m_buffers[index].GetMemory(desc);
}

} // namespace

int Context::CreateBuffer() {
    return m_bufferManager->CreateBuffer();
}

void Context::ResetBuffer(int index) {
    m_bufferManager->ResetBuffer(index);
}

void Context::DeleteBuffer(int index) {
    m_bufferManager->DeleteBuffer(index);
}

void Context::SetBuffer(int index, size_t addr) {
    m_bufferIndex = index;
    m_bufferManager->SetBuffer(index, addr);
}

void Context::CreateBufferManager() {
    m_bufferManager = std::make_unique<BufferManagerImpl>(this);
}

} // namespace base
} // namespace onednn
} // namespace arhat

