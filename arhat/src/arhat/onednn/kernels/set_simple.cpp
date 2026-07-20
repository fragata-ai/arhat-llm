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

#include "arhat/onednn/kernels/code.hpp"

namespace arhat {
namespace onednn {
namespace kernels {

namespace {

const char g_kernelCodeSet[] = R"(
kernel void set_simple(const global DATA_T *src, global DATA_T *dst) {
    src += SRC_BASE;
    dst += DST_BASE;

    int i0 = get_group_id(2);
    int i1 = get_group_id(1);
    int i2 = get_group_id(0);

    const global DATA_T *src_ptr = src + i0 * SRC_S0 + i1 * SRC_S1 + i2 * SRC_S2;
    global DATA_T *dst_ptr = dst + POFFS + i0 * PS0 + i1 * PS1 + i2 * PS2;

    for (int i3 = get_local_id(0); i3 < SRC_D3; i3 += get_local_size(0)) {
        dst_ptr[i3 * PS3] = src_ptr[i3 * SRC_S3];
    }
} 

)";

} // namespace

const char *SetSimpleKernelCode() {
    return g_kernelCodeSet;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

