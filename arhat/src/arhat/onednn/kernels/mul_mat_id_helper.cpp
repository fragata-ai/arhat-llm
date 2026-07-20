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

const char g_kernelCodeMulMatIdHelper[] = R"(
// To reduce shared memory use, store "it" and "iex_used" with 22 / 10 bits each. 

inline uint make_store(const uint it, const uint iex_used) {
    return (it & 0x003FFFFF) | (iex_used << 22);
}

inline uint store_get_it(uint data) {
    return data & 0x003FFFFF;
}

inline uint store_get_iex_used(uint data) {
    return data >> 22; 
}

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get
//     a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored
//     in expert_bounds[i : i + 1].

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
kernel void mm_ids_helper(
        const global int *ids, 
        global int *ids_src1, 
        global int *ids_dst, 
        global int *expert_bounds,
        const int n_tokens, 
        const int n_expert_used_var, 
        const int nchannels_y, 
        const int si1, 
        const int sis1
        SHAPE_INFO_ARGS) {

    ids += IDS_BASE;
    ids_src1 += IDS_SRC1_BASE;
    ids_dst += IDS_DST_BASE;
    expert_bounds += EXPERT_BOUNDS_BASE;

    const int n_expert_used = (N_EXPERT_USED == 0) ? n_expert_used_var : N_EXPERT_USED; 
    const int expert = GID_0;

    local uint store[NE_STORE];

    int nex_prev = 0;   // Number of columns for experts with a lower index
    int it_compact = 0; // Running index for the compact slice of this expert

    if (N_EXPERT_USED == 0) {
        // Generic implementation
        for (int it = 0; it < n_tokens; it++) {
            // The index at which the expert is used, if any
            int iex_used = -1;
            for (int iex = LID_0; iex < n_expert_used; iex += SG_SIZE) {
                const int expert_used = ids[it * si1 + iex];
                nex_prev += (expert_used < expert);
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = make_store(it, iex_used);
            }

            if (sub_group_any(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used
        for (int it0 = 0; it0 < n_tokens; it0 += SG_SIZE / NEU_PADDED) {
            const int it = it0 + LID_0 / NEU_PADDED;

            // The index at which the expert is used, if any
            const int iex = LID_0 % NEU_PADDED;
            const int expert_used = 
                ((NEU_PADDED == N_EXPERT_USED || iex < N_EXPERT_USED) && it < n_tokens) ?
                    ids[it * si1 + iex] : INT_MAX;
            const int iex_used = (expert_used == expert) ? iex : -1;
            nex_prev += (expert_used < expert);

            // Whether the threads at this token position have used the expert

            const int it_compact_add_self = 
                sub_group_clustered_reduce_logical_or((iex_used != -1), NEU_PADDED);

            // Do a scan over threads at lower token positions in subgroup
            // to get the correct index for writing data
            int it_compact_add_lower = 0;
            unroll_for (int offset = NEU_PADDED; offset < SG_SIZE; offset += NEU_PADDED) {
                const int tmp = sub_group_shuffle_up(it_compact_add_self, offset);
                if (LID_0 >= (uint)offset) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = make_store(it, iex_used);
            }

            // The thread with the highest index in the subgroup always has the sum over
            // the whole subgroup, use it to increment all threads
            it_compact += sub_group_shuffle(it_compact_add_lower + it_compact_add_self, SG_SIZE - 1);
        }
    }

    nex_prev = sub_group_reduce_add(nex_prev);

    for (int itc = LID_0; itc < it_compact; itc += SG_SIZE) {
        const uint store_it = store[itc];
        const int it = store_get_it(store_it);
        const int iex_used = store_get_iex_used(store_it);
        ids_src1[nex_prev + itc] = it * sis1 + iex_used % nchannels_y;
        ids_dst[nex_prev + itc] = it * n_expert_used + iex_used;
    }

    if (LID_0 != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < (int)GDIM_0 - 1) {
        return;
    }

    expert_bounds[GDIM_0] = nex_prev + it_compact;
}

)";

} // namespace

const char *MulMatIdHelperKernelCode() {
    return g_kernelCodeMulMatIdHelper;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

