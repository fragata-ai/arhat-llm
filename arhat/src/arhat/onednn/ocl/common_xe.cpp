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

#include <ostream>

#include "arhat/onednn/ocl/common_xe.hpp"

namespace arhat {
namespace onednn {
namespace ocl {

namespace {

const char g_grid[] = R"(
#define GDIM_0 get_num_groups(0)
#define GDIM_1 get_num_groups(1)
#define GDIM_2 get_num_groups(2)

#define GID_0 get_group_id(0)
#define GID_1 get_group_id(1)
#define GID_2 get_group_id(2)

#define LDIM_0 get_local_size(0)
#define LDIM_1 get_local_size(1)
#define LDIM_2 get_local_size(2)

#define LID_0 get_local_id(0)
#define LID_1 get_local_id(1)
#define LID_2 get_local_id(2)

)";

const char g_unroll[] = R"(
#define unroll_for __attribute__((opencl_unroll_hint)) for

)";

const char g_copy[] = R"(
#define COPY_US(ADST, ASRC, dst, src) *(ADST ushort *)(dst) = *(const ASRC ushort *)(src)
#define COPY_UI(ADST, ASRC, dst, src) *(ADST uint *)(dst) = *(const ASRC uint *)(src)
#define COPY_UL(ADST, ASRC, dst, src) *(ADST ulong *)(dst) = *(const ASRC ulong *)(src)

#define COPY_H_1(ADST, ASRC, dst, src) COPY_US(ADST, ASRC, dst, src)
#define COPY_H_2(ADST, ASRC, dst, src) COPY_UI(ADST, ASRC, dst, src)
#define COPY_H_4(ADST, ASRC, dst, src) COPY_UL(ADST, ASRC, dst, src)

#define COPY_H2_1(ADST, ASRC, dst, src) COPY_UI(ADST, ASRC, dst, src)
#define COPY_H2_2(ADST, ASRC, dst, src) COPY_UL(ADST, ASRC, dst, src)

#define COPY_F_1(ADST, ASRC, dst, src) COPY_UI(ADST, ASRC, dst, src)
#define COPY_F_2(ADST, ASRC, dst, src) COPY_UL(ADST, ASRC, dst, src)

#define COPY_F2_1(ADST, ASRC, dst, src) COPY_UL(ADST, ASRC, dst, src)

#define CALL_COPY(F, N, ADST, ASRC, dst, src) F##_##N(ADST, ASRC, dst, src)

#define COPY_RG_H(N, dst, src) CALL_COPY(COPY_H, N, private, global, dst, src)
#define COPY_GR_H(N, dst, src) CALL_COPY(COPY_H, N, global, private, dst, src)
#define COPY_RL_H(N, dst, src) CALL_COPY(COPY_H, N, private, local, dst, src)
#define COPY_LR_H(N, dst, src) CALL_COPY(COPY_H, N, local, private, dst, src)
#define COPY_LG_H(N, dst, src) CALL_COPY(COPY_H, N, local, global, dst, src)
#define COPY_GL_H(N, dst, src) CALL_COPY(COPY_H, N, global, local, dst, src)

#define COPY_RG_H2(N, dst, src) CALL_COPY(COPY_H2, N, private, global, dst, src)
#define COPY_GR_H2(N, dst, src) CALL_COPY(COPY_H2, N, global, private, dst, src)
#define COPY_RL_H2(N, dst, src) CALL_COPY(COPY_H2, N, private, local, dst, src)
#define COPY_LR_H2(N, dst, src) CALL_COPY(COPY_H2, N, local, private, dst, src)
#define COPY_LG_H2(N, dst, src) CALL_COPY(COPY_H2, N, local, global, dst, src)
#define COPY_GL_H2(N, dst, src) CALL_COPY(COPY_H2, N, global, local, dst, src)

#define COPY_RG_F(N, dst, src) CALL_COPY(COPY_F, N, private, global, dst, src)
#define COPY_GR_F(N, dst, src) CALL_COPY(COPY_F, N, global, private, dst, src)
#define COPY_RL_F(N, dst, src) CALL_COPY(COPY_F, N, private, local, dst, src)
#define COPY_LR_F(N, dst, src) CALL_COPY(COPY_F, N, local, private, dst, src)
#define COPY_LG_F(N, dst, src) CALL_COPY(COPY_F, N, local, global, dst, src)
#define COPY_GL_F(N, dst, src) CALL_COPY(COPY_F, N, global, local, dst, src)

#define COPY_RG_F2(N, dst, src) CALL_COPY(COPY_F2, N, private, global, dst, src)
#define COPY_GR_F2(N, dst, src) CALL_COPY(COPY_F2, N, global, private, dst, src)
#define COPY_RL_F2(N, dst, src) CALL_COPY(COPY_F2, N, private, local, dst, src)
#define COPY_LR_F2(N, dst, src) CALL_COPY(COPY_F2, N, local, private, dst, src)
#define COPY_LG_F2(N, dst, src) CALL_COPY(COPY_F2, N, local, global, dst, src)
#define COPY_GL_F2(N, dst, src) CALL_COPY(COPY_F2, N, global, local, dst, src)

)";

#if 0 // TODO: Revise this
const char g_imad[] = R"(
inline int imad(char4 a, char4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

)";

#else
const char g_imad[] = R"(
inline int imad(char4 a, char4 b, int c) {
    return dot_acc_sat(a, b, c);
}

)";
#endif

const char g_fastDiv[] = R"(
inline uint fastdiv(uint a, uint fd0, uint fd1) {
    const uint hi = mul_hi(a, fd0);
    return (hi + a) >> fd1;
}

inline uint fastmod(uint a, uint b, uint fd0, uint fd1) {
    return a - fastdiv(a, fd0, fd1) * b;
} 

)";

} // namespace

//
//    CommonXe
//

void CommonXe::EmitGrid(std::ostream &os) {
    os << g_grid;
}

void CommonXe::EmitUnroll(std::ostream &os) {
    os << g_unroll;
}

void CommonXe::EmitCopy(std::ostream &os) {
    os << g_copy;
}

void CommonXe::EmitImad(std::ostream &os) {
    os << g_imad;
}

void CommonXe::EmitFastDiv(std::ostream &os) {
    os << g_fastDiv;
}

} // namespace ocl
} // namespace onednn
} // namespace arhat

