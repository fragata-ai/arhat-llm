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

#include <memory>

#include "arhat/core/runtime.hpp"

#include "arhat/onednn/gpu/runtime.hpp"

namespace arhat {
namespace onednn {
namespace gpu {

class MulMatIdVec {
public:
    MulMatIdVec(Context *context);
    ~MulMatIdVec();
public:
    std::unique_ptr<core::Node> CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
private:
    Context *m_context;
};

class MulMatIdMm {
public:
    MulMatIdMm(Context *context);
    ~MulMatIdMm();
public:
    std::unique_ptr<core::Node> CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
private:
    Context *m_context;
};

class MulMatIdQuantSimple {
public:
    MulMatIdQuantSimple(Context *context);
    ~MulMatIdQuantSimple();
public:
    std::unique_ptr<core::Node> CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
private:
    Context *m_context;
};

class MulMatIdQuantVec {
public:
    MulMatIdQuantVec(Context *context);
    ~MulMatIdQuantVec();
public:
    std::unique_ptr<core::Node> CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
private:
    Context *m_context;
};

class MulMatIdQuantMm {
public:
    MulMatIdQuantMm(Context *context);
    ~MulMatIdQuantMm();
public:
    std::unique_ptr<core::Node> CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
private:
    Context *m_context;
};

class MulMatIdQuantVecV2 {
public:
    MulMatIdQuantVecV2(Context *context);
    ~MulMatIdQuantVecV2();
public:
    std::unique_ptr<core::Node> CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
private:
    Context *m_context;
};

class MulMatIdQuantVecV2Opt {
public:
    MulMatIdQuantVecV2Opt(Context *context);
    ~MulMatIdQuantVecV2Opt();
public:
    std::unique_ptr<core::Node> CreateNode(
        core::Node *a, 
        core::Node *b,
        core::Node *ids);
private:
    Context *m_context;
};

} // namespace gpu
} // namespace onednn
} // namespace arhat

