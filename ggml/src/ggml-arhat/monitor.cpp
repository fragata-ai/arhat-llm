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

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>

#include "ggml.h"
#include "ggml-impl.h"

#include "arhat/core/runtime.hpp"

#include "ggml-arhat/monitor.hpp"

namespace core = arhat::core;

namespace {

uint32_t F16ToBit32(uint16_t x) {
    uint32_t s = uint32_t(x & 0x8000) << 16;
    uint32_t e = uint32_t(x & 0x7c00) >> 10;
    uint32_t m = uint32_t(x & 0x03ff) << 13;
    if (e == 0x1f) {
        if (m == 0) {
            // Inf
            return (s | 0x7f800000 | m);
        } else {
            // NaN
            return (s | 0x7fc00000 | m);
        }
    }
    if (e == 0) {
        if (m == 0) {
            // zero
            return s;
        }
        // normalize
        e++;
        while ((m & 0x7f800000) == 0) {
            m <<= 1;
            e--;
        }
        m &= 0x007fffff;
    }
    return (s | ((e + (0x7f - 0xf)) << 23) | m);
}

float ToFloat(uint16_t x) {
    union {
        uint32_t i;
        float f;
    } u32;
    u32.i = F16ToBit32(x);
    return u32.f;
}

float ToFloat(float x) {
    return x;
}

float ToFloat(int32_t x) {
    return float(x);
}

struct MemoryStat {
    size_t volume = 0;
    double sum = 0.0;
    float fmin = 0.0f;
    size_t imin = 0;
    float fmax = 0.0f;
    size_t imax = 0;
};

template<typename T>
MemoryStat GetMemoryStat(const std::vector<T> &data) {
    MemoryStat stat;
    stat.volume = data.size();
    for (size_t i = 0; i < stat.volume; i++) {
        float x = ToFloat(data[i]);
        stat.sum += double(x);
        if (i == 0) {
            stat.fmin = x;
            stat.imin = 0;
            stat.fmax = x;
            stat.imax = 0;
        } else {
            if (x < stat.fmin) {
                stat.fmin = x;
                stat.imin = i;
            }
            if (x > stat.fmax) {
                stat.fmax = x;
                stat.imax = i;
            }
        }
    }
    return stat;
}

} // namespace

//
//    WallClock
//

WallClock::WallClock(): 
        m_elapsed(0.0f) { }

WallClock::~WallClock() { }

void WallClock::Reset() {
    m_elapsed = 0.0f;
}

void WallClock::Start() {
    start = std::chrono::steady_clock::now();
}

void WallClock::Stop() {
    end = std::chrono::steady_clock::now();
    m_elapsed +=
        std::chrono::duration_cast<
            std::chrono::duration<float, std::milli>>(end - start).count();
}

float WallClock::Elapsed() {
    return m_elapsed;
}

//
//    ArhatComputeMonitor
//

ArhatComputeMonitor::ArhatComputeMonitor(core::Context *context):
        m_context(context),
        m_graphCount(0),
        m_enableGraph(false),
        m_enableNode(false),
        m_cgraph(nullptr),
        m_tensorIndex(0),
        m_tensor(nullptr),
        m_node(nullptr) { 
    InitConf();        
}

ArhatComputeMonitor::~ArhatComputeMonitor() { }

void ArhatComputeMonitor::StartGraph(ggml_cgraph *cgraph) {
    m_enableGraph = IsGraphEnabled();
    if (!m_enableGraph) {
        return;
    }
    m_cgraph = cgraph;
    printf("==== Start graph %d: n_nodes %d\n", m_graphCount, cgraph->n_nodes);
}

void ArhatComputeMonitor::EndGraph() {
    if (!m_enableGraph) {
        m_graphCount++;
        return;
    }
    printf("==== End graph %d\n", m_graphCount);
    m_cgraph = nullptr;
    m_graphCount++;
}

void ArhatComputeMonitor::StartNode(
        int index,
        ggml_tensor *tensor,
        core::Node *node) {
    m_enableNode = IsNodeEnabled(index, tensor, node);
    if (!m_enableNode) {
        return;
    }
    m_tensorIndex = index;
    m_tensor = tensor;
    m_node = node;
    printf("---- Node %d\n", m_tensorIndex);
    if (m_conf.printOp) {
        PrintOp();
        if (m_conf.printOpSrc) {
            PrintOpSrc();
        }
    }
    if (m_conf.printTime) {
        StartTimer();
    }
}

void ArhatComputeMonitor::EndNode() {
    if (!m_enableNode) {
        return;
    }
    if (m_conf.printTime) {
        EndTimer();
    }
    if (m_conf.printMemoryStat) {
        PrintMemoryStat();
    }
    m_tensorIndex = 0;
    m_tensor = nullptr;
    m_node = nullptr;
}

void ArhatComputeMonitor::InitConf() {
    // TODO: Make configurable (e.g. using environment variables)
    m_conf.enable = false;
    m_conf.graphCountStart = 2;
    m_conf.graphCountStop = 5;
    m_conf.printOp = true;
    m_conf.printOpSrc = true;
    m_conf.printTime = true;
    m_conf.printMemoryStat = true;
    m_conf.opSet.insert(GGML_OP_MUL_MAT);
    m_conf.opSet.insert(GGML_OP_FLASH_ATTN_EXT);
}

bool ArhatComputeMonitor::IsGraphEnabled() {
    if (!m_conf.enable) {
        return false;
    }
    if (m_graphCount < m_conf.graphCountStart || m_graphCount >= m_conf.graphCountStop) {
        return false;
    }
    return true;
}

bool ArhatComputeMonitor::IsNodeEnabled(
        int index,
        ggml_tensor *tensor,
        core::Node *node) {
    if (!m_enableGraph) {
        return false;
    }
    if (!FindOp(tensor)) {
        return false;
    }
    return true;
}

bool ArhatComputeMonitor::FindOp(ggml_tensor *tensor) {
    if (m_conf.opSet.empty()) {
        return true;
    }
    if (m_conf.opSet.count(tensor->op) != 0) {
        return true;
    }
    return false;
}

void ArhatComputeMonitor::PrintOp() {
    PrintTensor("[DST]", m_tensor);
}

void ArhatComputeMonitor::PrintOpSrc() {
    char tag[64];
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (m_tensor->src[i] != nullptr) {
            snprintf(tag, 64, "[SRC%d]", i);
            PrintTensor(tag, m_tensor->src[i]);
        }
    }
}

void ArhatComputeMonitor::PrintTensor(const char *tag, ggml_tensor *tensor) {
    const char *opName = ggml_op_name(tensor->op);
    const char *typeName = ggml_type_name(tensor->type);
    size_t ne0 = size_t(tensor->ne[0]);
    size_t ne1 = size_t(tensor->ne[1]);
    size_t ne2 = size_t(tensor->ne[2]);
    size_t ne3 = size_t(tensor->ne[3]);
    size_t nb0 = size_t(tensor->nb[0]);
    size_t nb1 = size_t(tensor->nb[1]);
    size_t nb2 = size_t(tensor->nb[2]);
    size_t nb3 = size_t(tensor->nb[3]);
    size_t offset = size_t(tensor->view_offs);
    printf("%s %s type %s ne [%zd %zd %zd %zd] nb [%zd %zd %zd %zd] offs %zd\n",
        tag, opName, typeName, ne3, ne2, ne1, ne0, nb3, nb2, nb1, nb0, offset);
}

void ArhatComputeMonitor::StartTimer() {
    m_context->Wait();
    m_timer.Reset();
    m_timer.Start();
}

void ArhatComputeMonitor::EndTimer() {
    m_context->Wait();
    m_timer.Stop();
    printf("[TIME] elapsed %g ms\n", m_timer.Elapsed());
}

void ArhatComputeMonitor::PrintMemoryStat() {
    ggml_type type = m_tensor->type;
    const int64_t *ne = m_tensor->ne;
    int volume = int(ne[0] * ne[1] * ne[2] * ne[3]);
    if (type != GGML_TYPE_F16 &&
            type != GGML_TYPE_F32 &&
            type != GGML_TYPE_I32) {
        // unsupported type: silently return
        return;
    }
    MemoryStat stat;
    if (type == GGML_TYPE_F16) {
        std::vector<uint16_t> data(volume);
        m_node->Read(data.data(), 0, int(volume * sizeof(uint16_t)));
        stat = GetMemoryStat(data);
    } else if (type == GGML_TYPE_F32) {
        std::vector<float> data(volume);
        m_node->Read(data.data(), 0, int(volume * sizeof(float)));
        stat = GetMemoryStat(data);        
    } else if (type == GGML_TYPE_I32) {
        std::vector<int32_t> data(volume);
        m_node->Read(data.data(), 0, int(volume * sizeof(int32_t)));
        stat = GetMemoryStat(data);        
    }
    printf("[STAT] volume %zd sum %g min %g (%zd) max %g (%zd)\n", 
        stat.volume, stat.sum, stat.fmin, stat.imin, stat.fmax, stat.imax);
}

