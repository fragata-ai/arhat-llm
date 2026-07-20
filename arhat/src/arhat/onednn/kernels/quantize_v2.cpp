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

const char g_kernelCodeQuantize_Q8_1_X4[] = R"(
inline void quantize(
        const global float4 *data_a,
        global block_q8_1_x4 *data_b,
        const uint ne,
        const uint num_blocks,
        const uint wgid) {

    const uint tid = get_sub_group_local_id();

    // Each thread handles a vec4, so 8 threads handle a block
    const uint blocks_per_group = SG_SIZE / 8;

    const uint block_in_wg = tid / 8;

    const uint ib = wgid * blocks_per_group + block_in_wg;
    const uint iqs = tid % 8;

    const uint ibx4_outer = ib / 4;
    const uint ibx4_inner = ib % 4;

    const uint required_x4_blocks = (ne + 127) / 128;
    if (ibx4_outer >= required_x4_blocks) {
        return;
    }

    const uint a_idx = ib * 8 + iqs;

    float4 vals = (a_idx < ne / 4) ? data_a[a_idx] : (float4)(0.0f);
    const float4 abs_vals = fabs(vals);

    // Find absolute max for each block
    const float thread_max = fmax(fmax(abs_vals.x, abs_vals.y), fmax(abs_vals.z, abs_vals.w));
    const float amax = sub_group_clustered_reduce_max(thread_max, 8);

    const float d = amax / 127.0f;
    const float d_inv = (d != 0.0f) ? 1.0f / d : 0.0f;
    vals = round(vals * d_inv);

    data_b[ibx4_outer].qs[ibx4_inner * 8 + iqs] = as_int(convert_char4(vals));

    // Calculate the sum for each block
    const float thread_sum = vals.x + vals.y + vals.z + vals.w;
    const float sum = sub_group_clustered_reduce_add(thread_sum, 8);
    if (iqs == 0) {
        data_b[ibx4_outer].ds[ibx4_inner] = convert_half2((float2)(d, sum * d));
    }
}

__attribute__((intel_reqd_sub_group_size(SG_SIZE))) 
kernel void quantize_q8_1_x4(
        const global float4 *data_a,
        global block_q8_1_x4 *data_b,
        const uint ne,
        const uint num_blocks
        SHAPE_INFO_ARGS) {

    data_a += A_BASE / sizeof(float4);
    data_b += B_BASE / sizeof(block_q8_1_x4);

    uint wgid = GID_0;
    while (wgid < num_blocks) {
        quantize(data_a, data_b, ne, num_blocks, wgid);
        wgid += GDIM_0;
    }
} 

)";

} // namespace

const char *QuantizeV2_Q8_1_X4_KernelCode() {
    return g_kernelCodeQuantize_Q8_1_X4;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

