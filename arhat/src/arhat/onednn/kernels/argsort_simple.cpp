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

const char g_kernelCodeArgSort[] = R"(
#define ORDER_ASC 0
#define ORDER_DESC 1

#define SWAP(x, y) { int tmp = (x); (x) = (y); (y) = tmp; } 

kernel void argsort_simple(
        const global float *src,
        global int *dst) {
    src += SRC_BASE;
    dst += DST_BASE;

    local int dst_row[SRC_D3_PAD];

    // bitonic sort
    int col = get_local_id(0);
    int row = get_group_id(1);

    if (col >= SRC_D3_PAD) {
        return;
    }

    const global float *x_row = src + row * SRC_D3;

    // initialize indices
    dst_row[col] = col;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= SRC_D3_PAD; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= SRC_D3 ||
                            (dst_row[ixj] < SRC_D3 && 
                                ((ORDER == ORDER_ASC) ?
                                    (x_row[dst_row[col]] > x_row[dst_row[ixj]]) :
                                    (x_row[dst_row[col]] < x_row[dst_row[ixj]])))) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= SRC_D3 ||
                            (dst_row[col] < SRC_D3 && 
                                ((ORDER == ORDER_ASC) ?
                                    (x_row[dst_row[col]] < x_row[dst_row[ixj]]) :
                                    (x_row[dst_row[col]] > x_row[dst_row[ixj]])))) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // copy result to dst without the padding
    if (col < SRC_D3) {
        dst[row * SRC_D3 + col] = dst_row[col];
    }
} 

)";

} // namespace

const char *ArgSortSimpleKernelCode() {
    return g_kernelCodeArgSort;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

