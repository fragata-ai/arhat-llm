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

const char g_kernelCodeTri[] = R"(
kernel void tri_simple(const global float *src, global float *dst) {
    src += SRC_BASE;
    dst += DST_BASE;

    int idx = get_global_id(0);
    if (idx >= N) {
        return;
    }

    int i0 = idx % DST_D3;
    int i1 = (idx / DST_D3) % DST_D2;

    int keep = 0;
    if (MODE == 0) {
        keep = (i0 >= i1);
    } else if (MODE == 1) {
        keep = (i0 > i1);
    } else if (MODE == 2) {
        keep = (i0 <= i1);
    } else {
        keep = (i0 < i1); 
    }

    dst[idx] = keep ? src[idx] : 0.0f; 
}

)";

} // namespace

const char *TriSimpleKernelCode() {
    return g_kernelCodeTri;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

