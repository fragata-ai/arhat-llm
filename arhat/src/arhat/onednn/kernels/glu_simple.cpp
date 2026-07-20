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

const char g_codeSwigluOaiOp[] = R"(
inline float op_swiglu_oai(float x, float g, float alpha, float limit) {
    x = fmin(x, limit);
    g = fmax(fmin(g, limit), -limit);
    float out_glu = x / (1.0f + exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
} 

)";

const char g_kernelCodeGlu[] = R"(
kernel void glu(
        const global T *x, 
        const global T *g, 
        global T *y, 
        const long k, 
        const long n, 
        const long o0, 
        const long o1
        SHAPE_INFO_ARGS) {

    x += X_BASE;
    g += G_BASE;
    y += Y_BASE;

    const long i = (long)LDIM_0 * GID_0 + LID_0;

    if (i >= k) {
        return;
    }

    const long j0 = (i / n) * o0 + i % n;
    const long j1 = (o0 == o1) ? j0 : (i / n) * o1 + i % n;

    y[i] = (T)(OP((float)x[j0]) * (float)g[j1]);
} 

)";

const char g_kernelCodeSwigluOai[] = R"(
kernel void swiglu_oai(
        const global T *x, 
        const global T *g, 
        global T *y, 
        const long k, 
        const long n, 
        const long o0, 
        const long o1, 
        float alpha, 
        float limit
        SHAPE_INFO_ARGS) {

    x += X_BASE;
    g += G_BASE;
    y += Y_BASE;

    const long i = (long)LDIM_0 * GID_0 + LID_0;

    if (i >= k) {
        return;
    }

    const long j0 = (i / n) * o0 + i % n;
    const long j1 = (o0 == o1) ? j0 : (i / n) * o1 + i % n;

    float xi = x[j0];
    float gi = g[j1];

    y[i] = op_swiglu_oai(xi, gi, alpha, limit);
} 

)";

} // namespace

const char *GluSimpleSwigluOaiOpCode() {
    return g_codeSwigluOaiOp;
}

const char *GluSimpleKernelCode() {
    return g_kernelCodeGlu;
}

const char *GluSimpleSwigluOaiKernelCode() {
    return g_kernelCodeSwigluOai;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

