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

const char g_codeSqrOp[] = R"(
inline float op_sqr(float x) {
    return x * x;
}

)";

const char g_codeSqrtOp[] = R"(
float op_sqrt(float x) {
    return sqrt(x);
}

)";

const char g_codeLogOp[] = R"(
inline float op_log(float x) {
    return log(x);
}

)";

const char g_codeSinOp[] = R"(
inline float op_sin(float x) {
    return sin(x);
}

)";

const char g_codeCosOp[] = R"(
inline float op_cos(float x) {
    return cos(x);
}

)";

const char g_codeAbsOp[] = R"(
inline float op_abs(float x) {
    return fabs(x);
}

)";

const char g_codeSgnOp[] = R"(
inline float op_sgn(float x) {
    return (x > 0.0f) ? 1.0f : (x < 0.0f) ? -1.0f : 0.0f;
}

)";

const char g_codeNegOp[] = R"(
inline float op_neg(float x) {
    return -x;
}

)";

const char g_codeStepOp[] = R"(
inline float op_step(float x) {
    return (x > 0.0f);
}

)";

const char g_codeTanhOp[] = R"(
inline float op_tanh(float x) {
    return tanh(x);
}

)";

const char g_codeEluOp[] = R"(
inline float op_elu(float x) {
    return (x > 0.0f) ? x : expm1(x);
}

)";

const char g_codeReluOp[] = R"(
inline float op_relu(float x) {
    return fmax(x, 0.0f);
}

)";

const char g_codeSigmoidOp[] = R"(
inline float op_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

)";

const char g_codeGeluOp[] = R"(
inline float op_gelu(float x) {
    const float GELU_COEF_A = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x))); 
}

)";

const char g_codeGeluQuickOp[] = R"(
inline float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;
    return x * (1.0f / (1.0f + exp(GELU_QUICK_COEF * x)));
}

)";

const char g_codeSiluOp[] = R"(
inline float op_silu(float x) {
    return x / (1.0f + exp(-x)); 
}

)";

const char g_codeHardswishOp[] = R"(
inline float op_hardswish(float x) {
    return x * fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f));
}

)";

const char g_codeHardsigmoidOp[] = R"(
inline float op_hardsigmoid(float x) {
    return fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f));
}

)";

const char g_codeExpOp[] = R"(
inline float op_exp(float x) {
    return exp(x);
}

)";

const char g_codeExpm1Op[] = R"(
inline float op_expm1(float x) {
    return expm1(x);
}

)";

const char g_codeSoftplusOp[] = R"(
inline float op_softplus(float x) {
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

)";

const char g_codeGeluErfOp[] = R"(
inline float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    return 0.5f * x * (1.0f + erf(x * SQRT_2_INV));
}

)";

const char g_codeFloorOp[] = R"(
inline float op_floor(float x) {
    return floor(x);
}

)";

const char g_codeCeilOp[] = R"(
inline float op_ceil(float x) {
    return ceil(x);
}

)";

const char g_codeRoundOp[] = R"(
inline float op_round(float x) {
    return round(x);
}

)";

const char g_codeTruncOp[] = R"(
inline float op_trunc(float x) {
    return trunc(x);
} 

)";

const char g_kernelCodeUnary[] = R"(
kernel void unary(
        const global T *x, 
        global T *y, 
        const int k
        SHAPE_INFO_ARGS) {

    x += X_BASE;
    y += Y_BASE;

    const int i = LDIM_0 * GID_0 + LID_0;

    if (i >= k) {
        return;
    }

    y[i] = (T)OP((float)x[i]);
} 

)";

const char g_kernelCodeXielu[] = R"(
kernel void xielu(
        const global T *x, 
        global T *y, 
        const int k, 
        float alpha_n, 
        float alpha_p, 
        float beta, 
        float eps
        SHAPE_INFO_ARGS) {

    x += X_BASE;
    y += Y_BASE;

    const int i = LDIM_0 * GID_0 + LID_0;

    if (i >= k) {
        return;
    }

    const float xi = (float)x[i];

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fmin(xi, eps);
    const float y_neg = (expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    y[i] = (T)out;
} 

)";

} // namespace

const char *UnarySimpleSqrOpCode() {
    return g_codeSqrOp;
}

const char *UnarySimpleSqrtOpCode() {
    return g_codeSqrtOp;
}

const char *UnarySimpleLogOpCode() {
    return g_codeLogOp;
}

const char *UnarySimpleSinOpCode() {
    return g_codeSinOp;
}

const char *UnarySimpleCosOpCode() {
    return g_codeCosOp;
}

const char *UnarySimpleAbsOpCode() {
    return g_codeAbsOp;
}

const char *UnarySimpleSgnOpCode() {
    return g_codeSgnOp;
}

const char *UnarySimpleNegOpCode() {
    return g_codeNegOp;
}

const char *UnarySimpleStepOpCode() {
    return g_codeStepOp;
}

const char *UnarySimpleTanhOpCode() {
    return g_codeTanhOp;
}

const char *UnarySimpleEluOpCode() {
    return g_codeEluOp;
}

const char *UnarySimpleReluOpCode() {
    return g_codeReluOp;
}

const char *UnarySimpleSigmoidOpCode() {
    return g_codeSigmoidOp;
}

const char *UnarySimpleGeluOpCode() {
    return g_codeGeluOp;
}

const char *UnarySimpleGeluQuickOpCode() {
    return g_codeGeluQuickOp;
}

const char *UnarySimpleSiluOpCode() {
    return g_codeSiluOp;
}

const char *UnarySimpleHardswishOpCode() {
    return g_codeHardswishOp;
}

const char *UnarySimpleHardsigmoidOpCode() {
    return g_codeHardsigmoidOp;
}

const char *UnarySimpleExpOpCode() {
    return g_codeExpOp;
}

const char *UnarySimpleExpm1OpCode() {
    return g_codeExpm1Op;
}

const char *UnarySimpleSoftplusOpCode() {
    return g_codeSoftplusOp;
}

const char *UnarySimpleGeluErfOpCode() {
    return g_codeGeluErfOp;
}

const char *UnarySimpleFloorOpCode() {
    return g_codeFloorOp;
}

const char *UnarySimpleCeilOpCode() {
    return g_codeCeilOp;
}

const char *UnarySimpleRoundOpCode() {
    return g_codeRoundOp;
}

const char *UnarySimpleTruncOpCode() {
    return g_codeTruncOp;
}

const char *UnarySimpleKernelCode() {
    return g_kernelCodeUnary;
}

const char *UnarySimpleXieluKernelCode() {
    return g_kernelCodeXielu;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

