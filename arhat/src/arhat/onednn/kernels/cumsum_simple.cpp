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

const char g_kernelCodeCumSumBlk[] = R"(
// max workgroup size is usually 1024, this covers various subgroups sizes
#define MAX_SUBGROUPS 128 

__attribute__((intel_reqd_sub_group_size(32)))
kernel void cumsum_simple_blk(
        const global float *src,
        global float *tmp,
        global float *dst) {
    src += SRC_BASE;
    dst += DST_BASE;

    const int i0 = get_group_id(2);
    const int i1 = get_group_id(1);
    const int i23 = get_group_id(0);

    const int nth = get_local_size(0);
    const int tid = get_local_id(0);

    const uint sg_size = get_sub_group_size();
    const uint sg_id = get_sub_group_id();
    const uint sg_lid = get_sub_group_local_id();

    const int ib = i23 / DST_D2;
    const int i3 = ib * nth;
    const int i2 = i23 % DST_D2;

    const global float *src_row = src + i0 * SRC_S0 + i1 * SRC_S1 + i2 * SRC_S2;
    global float *tmp_row = tmp + i0 * TMP_S0 + i1 * TMP_S1 + i2 * TMP_S2;
    global float *dst_row = dst + i0 * DST_S0 + i1 * DST_S1 + i2 * DST_S2;

    __local float partial[MAX_SUBGROUPS];

    float v = 0.0f;
    if (i3 + tid < DST_D3) {
        v = src_row[i3 + tid];
    }

    float s = sub_group_scan_inclusive_add(v);
    if (sg_lid == sg_size - 1) {
        partial[sg_id] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // NB: subgroup size should be larger than number of subgroups
    // assuming max workgroup size of 1024, subgroup size should be >= 32
    if (sg_id == 0) {
        float x = 0.0f;
        if (sg_lid < get_num_sub_groups()) {
            x = partial[sg_lid];
        }
        float ex = sub_group_scan_exclusive_add(x);
        if (sg_lid < get_num_sub_groups()) {
            partial[sg_lid] = ex;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    s += partial[sg_id];

    if (i3 + tid < DST_D3) {
        dst_row[i3 + tid] = s;
    }
    if (nth < DST_D3 && tid == nth - 1) {
        tmp_row[ib] = s;
    }
}

)";

const char g_kernelCodeCumSumAdd[] = R"(
kernel void cumsum_simple_add(const global float *tmp, global float *dst) {
    dst += DST_BASE;

    const int i0 = get_group_id(2);
    const int i1 = get_group_id(1);
    const int i23 = get_group_id(0);

    const int nth = get_local_size(0);
    const int tid = get_local_id(0);

    const int ib = i23 / DST_D2;
    if (ib == 0) {
        return;
    }
    const int i3 = ib * nth;
    const int i2 = i23 % DST_D2;

    const global float *tmp_row = tmp + i0 * TMP_S0 + i1 * TMP_S1 + i2 * TMP_S2;
    global float *dst_row  = dst + i0 * DST_S0 + i1 * DST_S1 + i2 * DST_S2;

    if (i3 + tid < DST_D3) {
        dst_row[i3 + tid] += tmp_row[ib - 1];
    }
} 

)";

} // namespace

const char *CumSumSimpleBlkKernelCode() {
    return g_kernelCodeCumSumBlk;
}

const char *CumSumSimpleAddKernelCode() {
    return g_kernelCodeCumSumAdd;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

