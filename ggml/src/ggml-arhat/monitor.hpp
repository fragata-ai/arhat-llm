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

#include <chrono>
#include <set>

#include "ggml.h"

#include "arhat/core/runtime.hpp"

namespace core = arhat::core;

class WallClock {
public:
    WallClock();
    ~WallClock();
public:
    void Reset();
    void Start();
    void Stop();
    float Elapsed();
private:
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    float m_elapsed;
};

//
//    ArhatComputeMonitorConf
//

struct ArhatComputeMonitorConf {
    bool enable = false;
    int graphCountStart = 0;
    int graphCountStop = 0;
    bool printOp = false;
    bool printOpSrc = false;
    bool printTime = false;
    bool printMemoryStat = false;
    std::set<ggml_op> opSet;
};

//
//    ArhatComputeMonitor
//

class ArhatComputeMonitor {
public:
    ArhatComputeMonitor(core::Context *context);
    ~ArhatComputeMonitor();
public:
    void StartGraph(ggml_cgraph *cgraph);
    void EndGraph();
    void StartNode(
        int index,
        ggml_tensor *tensor,
        core::Node *node);
    void EndNode();
private:
    void InitConf();
    bool IsGraphEnabled();
    bool IsNodeEnabled(
        int index,
        ggml_tensor *tensor,
        core::Node *node);
    bool FindOp(ggml_tensor *tensor);
    void PrintOp();
    void PrintOpSrc();
    void PrintTensor(const char *tag, ggml_tensor *tensor);
    void StartTimer();
    void EndTimer();
    void PrintMemoryStat();
private:
    core::Context *m_context;
    ArhatComputeMonitorConf m_conf;
    WallClock m_timer;
    int m_graphCount;
    bool m_enableGraph;
    bool m_enableNode;
    ggml_cgraph *m_cgraph;
    int m_tensorIndex;
    ggml_tensor *m_tensor;
    core::Node *m_node;
};

