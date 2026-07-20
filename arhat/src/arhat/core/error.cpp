/* 
* MIT License
*
* Copyright (c) 2026 FRAGATA COMPUTER SYSTEMS AG
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
#include <cstdarg>
#include <string>

#include "arhat/core/runtime.hpp"

#define ENABLE_PRINT_MSG 1

namespace arhat {
namespace core {

//
//    RuntimeError
//

RuntimeError::RuntimeError(const char *msg):
        m_msg(msg) { }

RuntimeError::RuntimeError(const std::string &msg):
        m_msg(msg) { }

RuntimeError::~RuntimeError() { }

const char *RuntimeError::what() const noexcept {
    return m_msg.c_str();
}

//
//    Public functions
//

void Error(const char *fmt, ...) {
    char msg[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
#if ENABLE_PRINT_MSG
    printf("*** Error: %s\n", msg);
#endif
    throw RuntimeError(msg);
}

void NotImplemented(const char *what) {
    Error("Not implemented: %s", what);
}

} // namespace core
} // namespace arhat

