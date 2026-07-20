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

//
//    LoadTiles
//

const char g_codeLoadTiles_Q4_0[] = R"(
inline void load_tiles_q4_0(
        const global char *x, 
        local int *x_tile,
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + 2 * MMQ_TILE_NE_K);
#else
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + TXS_QS);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;
    const int kbx = txi / QI4_0;
    const int kqsx = txi % QI4_0;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q4_0 *bxi = (const global block_q4_0 *)x + kbx0 + i * stride + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + 0] = 
            ISUB_SAT((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + QI4_0] = 
            ISUB_SAT((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif
    }

    const int kbxd = LID_0 % BLOCKS_PER_TILE_X_ROW;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = i0 + LID_1 * ROWS_PER_SG + LID_0 / BLOCKS_PER_TILE_X_ROW;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q4_0 *bxi = (const global block_q4_0 *)x + kbx0 + i * stride + kbxd;

#ifdef USE_MMA
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = bxi->d;
#else
        x_df[i * (MMQ_TILE_NE_K / QI4_0) + i / QI4_0 + kbxd] = bxi->d;
#endif
    }
}

)";

const char g_codeLoadTiles_Q4_1[] = R"(
inline void load_tiles_q4_1(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + 2 * MMQ_TILE_NE_K);
#else
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + TXS_QS);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;
    const int kbx = txi / QI4_1;
    const int kqsx = txi % QI4_1;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q4_1 *bxi = (const global block_q4_1 *)x + kbx0 + i * stride + kbx;
        const int qs0 = get_int_b4(bxi->qs, kqsx);

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kbx * (2 * QI4_1) + kqsx + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kbx * (2 * QI4_1) + kqsx + QI4_1] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif
    }

    const int kbxd = LID_0 % BLOCKS_PER_TILE_X_ROW;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = i0 + LID_1 * ROWS_PER_SG + LID_0 / BLOCKS_PER_TILE_X_ROW;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q4_1 *bxi = (const global block_q4_1 *)x + kbx0 + i * stride + kbxd;

#ifdef USE_MMA
        x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + kbxd] = bxi->dm;
#else
        x_dm[i * (MMQ_TILE_NE_K / QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
#endif
    }
}

)";

const char g_codeLoadTiles_Q5_0[] = R"(
inline void load_tiles_q5_0(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + MMQ_TILE_NE_K * 2);
#else
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + TXS_QS);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;
    const int kbx = txi / QI5_0;
    const int kqsx = txi % QI5_0;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q5_0 *bxi = (const global block_q5_0 *)x + kbx0 + i * stride + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >> 0) & 0x0F0F0F0F;
        qs0 |= (qh << 4) & 0x00000010;   // 0 ->  4
        qs0 |= (qh << 11) & 0x00001000;  // 1 -> 12
        qs0 |= (qh << 18) & 0x00100000;  // 2 -> 20
        qs0 |= (qh << 25) & 0x10000000;  // 3 -> 28
        qs0 = ISUB_SAT(qs0, 0x10101010); // subtract 16

        int qs1 = (ql >> 4) & 0x0F0F0F0F;
        qs1 |= (qh >> 12) & 0x00000010;  // 16 ->  4
        qs1 |= (qh >> 5) & 0x00001000;   // 17 -> 12
        qs1 |= (qh << 2) & 0x00100000;   // 18 -> 20
        qs1 |= (qh << 9) & 0x10000000;   // 19 -> 28
        qs1 = ISUB_SAT(qs1, 0x10101010); // subtract 16

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI5_0) + kqsx + 0] = qs0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI5_0) + kqsx + QI5_0] = qs1;
#else
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kbx * (2 * QI5_0) + kqsx + 0] = qs0;
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kbx * (2 * QI5_0) + kqsx + QI5_0] = qs1;
#endif
    }

    const int kbxd = LID_0 % BLOCKS_PER_TILE_X_ROW;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = i0 + LID_1 * ROWS_PER_SG + LID_0 / BLOCKS_PER_TILE_X_ROW;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q5_0 *bxi = (const global block_q5_0 *)x + kbx0 + i * stride + kbxd;

#ifdef USE_MMA
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = bxi->d;
#else
        x_df[i * (MMQ_TILE_NE_K / QI5_0) + i / QI5_0 + kbxd] = bxi->d;
#endif
    }
} 

)";

const char g_codeLoadTiles_Q5_1[] = R"(
inline void load_tiles_q5_1(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + 2 * MMQ_TILE_NE_K);
#else
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + TXS_QS);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;
    const int kbx = txi / QI5_1;
    const int kqsx = txi % QI5_1;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q5_1 *bxi = (const global block_q5_1 *)x + kbx0 + i * stride + kbx;

        const int ql = get_int_b4(bxi->qs, kqsx);
        const int qh = get_int_b4(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >> 0) & 0x0F0F0F0F;
        qs0 |= (qh << 4) & 0x00000010;  // 0 ->  4
        qs0 |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0 |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0 |= (qh << 25) & 0x10000000; // 3 -> 28

        int qs1 = (ql >> 4) & 0x0F0F0F0F;
        qs1 |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1 |= (qh >> 5) & 0x00001000;  // 17 -> 12
        qs1 |= (qh << 2) & 0x00100000;  // 18 -> 20
        qs1 |= (qh << 9) & 0x10000000;  // 19 -> 28

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kbx * (2 * QI5_1) + kqsx + 0] = qs0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kbx * (2 * QI5_1) + kqsx + QI5_1] = qs1;
#else
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kbx * (2 * QI5_1) + kqsx + 0] = qs0;
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kbx * (2 * QI5_1) + kqsx + QI5_1] = qs1;
#endif
    }

    const int kbxd = LID_0 % BLOCKS_PER_TILE_X_ROW;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = i0 + LID_1 * ROWS_PER_SG + LID_0 / BLOCKS_PER_TILE_X_ROW;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q5_1 *bxi = (const global block_q5_1 *)x + kbx0 + i * stride + kbxd;

#ifdef USE_MMA
        x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + kbxd] = bxi->dm;
#else
        x_dm[i * (MMQ_TILE_NE_K / QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
#endif
    }
}

)";

const char g_codeLoadTiles_Q8_0[] = R"(
inline void load_tiles_q8_0(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_tile + 2 * MMQ_TILE_NE_K);
#else
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + TXS_QS);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;
    const int kbx = txi / QI8_0;
    const int kqsx = txi % QI8_0;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q8_0 *bxi = (const global block_q8_0 *)x + kbx0 + i * stride + kbx;

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 0 + txi] = get_int_b2(bxi[0].qs, kqsx);
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K / QI8_0].qs, kqsx);
#else
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + 0 + txi] = get_int_b2(bxi[0].qs, kqsx);
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K / QI8_0].qs, kqsx);
#endif
    }

    const int kbxd = LID_0 % BLOCKS_PER_TILE_X_ROW;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = i0 + LID_1 * ROWS_PER_SG + LID_0 / BLOCKS_PER_TILE_X_ROW;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q8_0 *bxi = (const global block_q8_0 *)x + kbx0 + i * stride + kbxd;

#ifdef USE_MMA
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = bxi->d;
#else
        x_df[i * (2 * MMQ_TILE_NE_K / QI8_0) + i / (QI8_0 / 2) + kbxd] = bxi->d;
#endif
    }
}

)";

const char g_codeLoadTiles_Q2_K[] = R"(
inline void load_tiles_q2_K(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + 2 * MMQ_TILE_NE_K);
#else
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + TXS_QS);
#endif

    const int kqsx = LID_0 % THREADS_PER_ROW;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + LID_1 * NROWS + LID_0 / THREADS_PER_ROW;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q2_K *bxi = (const global block_q2_K *)x + kbx0 + i * stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

        unroll_for (int l = 0; l < QR2_K; l++) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2 * l)) & 0x03030303;

#ifdef USE_MMA
            x_qs[i * MMQ_MMA_TILE_X_K_Q2_K + k] = x_qs_k;
#else
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif
        }

        const int sc_m = bxi->scales[kqsx];
        const half2 x_dm_ik = bxi->dm * (half2)(sc_m & 0x0F, sc_m >> 4);

#ifdef USE_MMA
        x_dm[i * MMQ_MMA_TILE_X_K_Q2_K + kqsx] = x_dm_ik;
#else
        x_dm[i * (MMQ_TILE_NE_K + 1) + kqsx] = x_dm_ik;
#endif
    }
} 

)";

const char g_codeLoadTiles_Q3_K[] = R"(
inline void load_tiles_q3_K(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + MMQ_TILE_NE_K * 2);
#else
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + TXS_QS);
    local int *x_sc = (local int *)(x_df + TXS_DM);
#endif

    const int kqsx = LID_0 % THREADS_PER_ROW;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + LID_1 * NROWS + LID_0 / THREADS_PER_ROW;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q3_K *bxi = (const global block_q3_K *)x + kbx0 + i * stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K / 2)) >> (4 * (kqsx / (QI3_K / 2)));

        unroll_for (int l = 0; l < QR3_K; l++) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;

            const int x_ql_k = (x_ql_0 >> (2 * l)) & 0x03030303;
            const int x_qh_k = ((x_qh_0 >> l)  << 2) & 0x04040404;

            const int x_qs_k = ISUB_SAT(x_ql_k | x_qh_k, 0x04040404);

#ifdef USE_MMA
            x_qs[i * MMQ_MMA_TILE_X_K_Q3_K + k] = x_qs_k;
#else
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif
        }
    }

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = i0 + LID_1 * ROWS_PER_SG + LID_0 / 4;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q3_K *bxi = (const global block_q3_K *)x + kbx0 + i * stride;

        const int ksc = LID_0 % 4;

        const int ksc_low = ksc % (QI3_K / 8);
        const int shift_low = 4 * (ksc / (QI3_K / 8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K / 8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = ISUB_SAT(sc_low | sc_high, 0x20202020);

#ifdef USE_MMA
        const char *sc8 = (const char *)&sc;
        const float d = bxi->d;

        unroll_for (int l = 0; l < int(sizeof(int)); l++) {
            x_df[i * MMQ_MMA_TILE_X_K_Q3_K + sizeof(int) * ksc + l] = d * sc8[l];
        }
#else
        x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + ksc] = sc;
#endif
    }

#ifndef USE_MMA
    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * SG_SIZE) {
        int i = (i0 + LID_1 * SG_SIZE + LID_0) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q3_K *bxi = (const global block_q3_K *)x + kbx0 + i * stride;

        x_df[i] = bxi->d;
    }
#endif
}

)";

const char *g_coreUnpackScales_Q45_K = R"(
inline int unpack_scales_q45_K(const global int *scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F) | // lower 4 bits
        ((scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030);                            // upper 2 bits
}

)";

const char g_codeLoadTiles_Q4_K[] = R"(
inline void load_tiles_q4_K(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + 2 * MMQ_TILE_NE_K);
#else
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + TXS_QS);
    local int *x_sc = (local int *)(x_dm + TXS_DM);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q4_K *bxi = (const global block_q4_K *)x + kbx0 + i * stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 16 * (txi / 8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + 16 * (txi / 8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i * (MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif
    }

#ifdef USE_MMA
    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        // SKIPPED: AMD-specific code

        int i = (i0 + LID_1 * ROWS_PER_SG + LID_0 / 2) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q4_K *bxi = (const global block_q4_K *)x + kbx0 + i * stride;

        const global int *scales = (const global int *)bxi->scales;
        const int ksc = LID_0 % 2;

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uchar *sc8 = (const uchar *)&sc32;
        const uchar *m8 = (const uchar *)&m32;

        const half2 dm = bxi->dm * (half2)(1.0f, -1.0f);

        unroll_for (int l = 0; l < sizeof(int); l++) {
            x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int) * ksc + l] = dm * (half2)(sc8[l], m8[l]);
        }
    }
#else
    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * SG_SIZE) {
        int i = (i0 + LID_1 * SG_SIZE + LID_0) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q4_K *bxi = (const global block_q4_K *)x + kbx0 + i * stride;

        x_dm[i] = bxi->dm;
    }
    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = (i0 + LID_1 * ROWS_PER_SG + LID_0 / (MMQ_TILE_NE_K / 8)) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q4_K *bxi = 
            (const global block_q4_K *)x + kbx0 + i * stride + (LID_0 % (MMQ_TILE_NE_K / 8)) / (QI4_K / 8);

        const global int *scales = (const global int *)bxi->scales;

        const int ksc = LID_0 % (MMQ_TILE_NE_K / 8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + ksc] = scales8;
    }
#endif
}

)";

const char g_codeLoadTiles_Q5_K[] = R"(
inline void load_tiles_q5_K(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + MMQ_TILE_NE_K * 2);
#else
    local int *x_qs = (local int *)x_tile;
    local half2 *x_dm = (local half2 *)(x_qs + TXS_QS);
    local int *x_sc = (local int *)(x_dm + TXS_DM);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q5_K *bxi = (const global block_q5_K *)x + kbx0 + i * stride;
        const int ky = QR5_K * txi;

        const int ql = get_int_b4(bxi->qs, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, txi % (QI5_K / 4));
        const int qh0 = ((qh >> (2 * (txi / (QI5_K / 4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (QI5_K / 4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K / 2) + txi % (QI5_K / 4) + 0;
        const int kq1 = ky - ky % (QI5_K / 2) + txi % (QI5_K / 4) + QI5_K / 4;

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kq0] = ql0 | qh0;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + kq1] = ql1 | qh1;
#else
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kq0] = ql0 | qh0;
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kq1] = ql1 | qh1;
#endif
    }

#ifdef USE_MMA
    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        // SKIPPED: AMD-specific code
        int i = (i0 + LID_1 * ROWS_PER_SG + LID_0 / 2) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q5_K *bxi = (const global block_q5_K *)x + kbx0 + i * stride;

        const global int *scales = (const global int *)bxi->scales;
        const int ksc = LID_0 % 2;

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

        const uchar *sc8 = (const uchar *)&sc32;
        const uchar *m8 = (const uchar *)&m32;

        const half2 dm = bxi->dm * (half2)(1.0f, -1.0f);

        unroll_for (int l = 0; l < int(sizeof(int)); l++) {
            x_dm[i * MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int) * ksc + l] = dm * (half2)(sc8[l], m8[l]);
        }
    }
#else
    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * SG_SIZE) {
        int i = (i0 + LID_1 * SG_SIZE + LID_0) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q5_K *bxi = (const global block_q5_K *)x + kbx0 + i * stride;

        x_dm[i] = bxi->dm;
    }

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = (i0 + LID_1 * ROWS_PER_SG + LID_0 / (MMQ_TILE_NE_K / 8)) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q5_K *bxi = (const global block_q5_K *)x + kbx0 + i * stride;

        const global int *scales = (const global int *)bxi->scales;

        const int ksc = LID_0 % (MMQ_TILE_NE_K / 8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + ksc] = scales8;
    }
#endif
}

)";

const char g_codeLoadTiles_Q6_K[] = R"(
inline void load_tiles_q6_K(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + MMQ_TILE_NE_K * 2);
    local int *x_sc = (local int *)(x_df + MMQ_TILE_NE_K / QI6_K);
#else
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + TXS_QS);
    local int *x_sc = (local int *)(x_df + TXS_DM);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q6_K *bxi = (const global block_q6_K *)x + kbx0 + i * stride;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K / 4) * (txi / (QI6_K / 2)) + txi % (QI6_K / 4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 = (qh >> ((txi & 0x08) >> 2)) & 0x30303030;

        const int kq0 = 2 * txi - txi % (QI6_K / 2) + 0;
        const int kq1 = 2 * txi - txi % (QI6_K / 2) + QI6_K / 2;

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q6_K + kq0] = ISUB_SAT(ql0 | qh0, 0x20202020);
        x_qs[i * MMQ_MMA_TILE_X_K_Q6_K + kq1] = ISUB_SAT(ql1 | qh1, 0x20202020);
#else
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kq0] = ISUB_SAT(ql0 | qh0, 0x20202020);
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kq1] = ISUB_SAT(ql1 | qh1, 0x20202020);
#endif
    }

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * SG_SIZE) {
        int i = (i0 + LID_1 * SG_SIZE + LID_0) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q6_K *bxi = (const global block_q6_K *)x + kbx0 + i * stride;

#ifdef USE_MMA
        x_df[i * MMQ_MMA_TILE_X_K_Q6_K] = bxi->d;
#else
        x_df[i * (MMQ_TILE_NE_K / QI6_K) + i / QI6_K] = bxi->d;
#endif
    }

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = (i0 + LID_1 * ROWS_PER_SG + LID_0 / (MMQ_TILE_NE_K / 8)) % MMQ_Y;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_q6_K *bxi = 
            (const global block_q6_K *)x + kbx0 + i * stride + (LID_0 % (MMQ_TILE_NE_K / 8)) / 4;

#ifdef USE_MMA
        x_sc[i * MMQ_MMA_TILE_X_K_Q6_K + LID_0 % 4] = get_int_b2(bxi->scales, LID_0 % (MMQ_TILE_NE_K / 8));
#else
        x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + LID_0 % (MMQ_TILE_NE_K / 8)] = get_int_b2(bxi->scales, LID_0 % (QI6_K / 8));
#endif
    }
}

)";

const char g_codeLoadTiles_Mxfp4[] = R"(
inline void load_tiles_mxfp4(
        const global char *x, 
        local int *x_tile, 
        const int kbx0, 
        const int i_max, 
        const int stride) {

#ifdef USE_MMA
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + MMQ_TILE_NE_K * 2);
#else
    local int *x_qs = (local int *)x_tile;
    local float *x_df = (local float *)(x_qs + TXS_QS);
#endif

    const int txi = (SG_SIZE > THREADS_PER_ROW) ? LID_0 % THREADS_PER_ROW : LID_0;
    const int kbx = txi / QI_MXFP4;
    const int kqsx = txi % QI_MXFP4;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NROWS * NUM_SGS) {
        int i = i0 + ((NROWS == 1) ? LID_1 : LID_1 * NROWS + LID_0 / THREADS_PER_ROW);

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_mxfp4 *bxi = (const global block_mxfp4 *)x + kbx0 + i * stride + kbx;

        const int aux_q4 = get_int_b1(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        const int k0 = kbx * (2 * QI_MXFP4) + kqsx;

#ifdef USE_MMA
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + k0 + 0] = v.x;
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_1 + k0 + QI_MXFP4] = v.y;
#else
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0 + 0] = v.x;
        x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0 + QI_MXFP4] = v.y;
#endif
    }

    const int kbxd = LID_0 % BLOCKS_PER_TILE_X_ROW;

    unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += NUM_SGS * ROWS_PER_SG) {
        int i = i0 + LID_1 * ROWS_PER_SG + LID_0 / BLOCKS_PER_TILE_X_ROW;

        if (NEED_CHECK) {
            i = min(i, i_max);
        }

        const global block_mxfp4 *bxi = (const global block_mxfp4 *)x + kbx0 + i * stride + kbxd;

#ifdef USE_MMA
        x_df[i * MMQ_MMA_TILE_X_K_Q8_1 + kbxd] = e8m0_to_fp32(bxi->e) * 0.5f;
#else
        x_df[i * (MMQ_TILE_NE_K / QI_MXFP4) + i / QI_MXFP4 + kbxd] = e8m0_to_fp32(bxi->e) * 0.5f;
#endif
    }
}

)";

//
//    VecDotDp4a
//

const char g_codeVecDotDp4a_Q4_0[] = R"(
inline void vec_dot_q4_0_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local float *x_df = (const local float *)x_qs + TXS_QS;
    const local int *y_qs = (const local int *)y + 4;
    const local half2 *y_ds = (const local half2 *)y;

// unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_0 * VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                const int kyqs = QI8_1 * ((k01 / 2) / (QI8_1 / 2)) + (k01 / 2) % (QI8_1 / 2);

                int u[2 * VDR_Q4_0_Q8_1_MMQ];

                unroll_for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; l++) {
                    u[2 * l + 0] = y_qs[j * MMQ_TILE_Y_K + kyqs + l];
                    u[2 * l + 1] = y_qs[j * MMQ_TILE_Y_K + kyqs + (l + QI4_0)];
                }

                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q4_0_q8_1_impl(
                        &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / QR4_0], 
                        u,
                        x_df[i * (MMQ_TILE_NE_K / QI4_0) + i / QI4_0 + k0 / (QR4_0 * QI4_0)], 
                        y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
} 

)";

const char g_codeVecDotDp4a_Q4_1[] = R"(
inline void vec_dot_q4_1_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local half2 *x_dm = (const local half2 *)x_qs + TXS_QS;
    const local int *y_qs = (const local int *)y + 4;
    const local half2 *y_ds = (const local half2 *)y;

// unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_1 * VDR_Q4_1_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                const int kyqs = QI8_1 * ((k01 / 2) / (QI8_1 / 2)) + (k01 / 2) % (QI8_1 / 2);

                int u[2 * VDR_Q4_1_Q8_1_MMQ];

                unroll_for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; l++) {
                    u[2 * l + 0] = y_qs[j * MMQ_TILE_Y_K + kyqs + l];
                    u[2 * l + 1] = y_qs[j * MMQ_TILE_Y_K + kyqs + (l + QI4_1)];
                }

                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q4_1_q8_1_impl(
                        &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / QR4_1], 
                        u,
                        x_dm[i * (MMQ_TILE_NE_K / QI4_1) + i / QI4_1 + k0 / (QR4_1 * QI4_1)], 
                        y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
} 

)";

const char g_codeVecDotDp4a_Q8_0[] = R"(
inline void vec_dot_q8_0_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local float *x_df = (const local float *)x_qs + TXS_QS;
    const local int *y_qs = (const local int *)y + 4;
    const local float *y_df = (const local float *)y;

// unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q8_0_q8_1_impl(
                        &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0], 
                        &y_qs[j * MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K],
                        x_df[i * (2 * MMQ_TILE_NE_K / QI8_0) + i / (QI8_0 / 2) + k0 / QI8_0], 
                        y_df[j * MMQ_TILE_Y_K + (k0 / QI8_1) % (MMQ_TILE_NE_K / QI8_1)]);
            }
        }
    }
} 

)";

const char g_codeVecDotDp4a_Q8_1[] = R"(
inline void vec_dot_q8_1_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local half2 *x_dm = (const local half2 *)x_qs + TXS_QS;
    const local int *y_qs = (const local int *)y + 4;
    const local half2 *y_ds = (const local half2 *)y;

// unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q8_1_q8_1_impl(
                        &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0], 
                        &y_qs[j * MMQ_TILE_Y_K + k01],
                        x_dm[i * (MMQ_TILE_NE_K / QI5_1) + i / QI5_1 + k0 / QI8_1], 
                        y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

)";

const char g_codeVecDotDp4a_Q2_K[] = R"(
inline void vec_dot_q2_K_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local half2 *x_dm = (const local half2 *)x_qs + TXS_QS;
    const local int *y_qs = (const local int *)y + 4;
    const local half2 *y_ds = (const local half2 *)y;

    float2 y_df[MMQ_X / NUM_SGS];
    unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
        const int j = j0 + LID_1;
        y_df[j0 / NUM_SGS] = convert_float2(y_ds[j * MMQ_TILE_Y_K]);
    }

    unroll_for (int k01 = 0; k01 < MMQ_TILE_NE_K / 2; k01 += QR2_K * VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                const int NS = 2;
                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q2_K_q8_1_impl_mmq(
                        NS,
                        &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0], 
                        &y_qs[j * MMQ_TILE_Y_K + k01],
                        &x_dm[i * (MMQ_TILE_NE_K + 1) + k0 / 4], 
                        (k01 < MMQ_TILE_NE_K / 2) ? y_df[j0 / NUM_SGS].x : y_df[j0 / NUM_SGS].y,
                        &y_ds[j * MMQ_TILE_Y_K + (1 + k01 / QI8_1)]);
            }
        }
    }

    // Some compilers fail to unroll the loop over k01 if
    // there is a conditional statement for ns in the inner loop.
    // As a workaround 2 separate loops are used instead.
    unroll_for (int k01 = MMQ_TILE_NE_K / 2; k01 < MMQ_TILE_NE_K; k01 += QR2_K * VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                const int NS = 1;
                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q2_K_q8_1_impl_mmq(
                        NS,
                        &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0], 
                        &y_qs[j * MMQ_TILE_Y_K + k01],
                        &x_dm[i * (MMQ_TILE_NE_K + 1) + k0 / 4], 
                        (k01 < MMQ_TILE_NE_K / 2) ? y_df[j0 / NUM_SGS].x : y_df[j0 / NUM_SGS].y,
                        &y_ds[j * MMQ_TILE_Y_K + (1 + k01 / QI8_1)]);
            }
        }
    }
}

)";

const char g_codeVecDotDp4a_Q3_K[] = R"(
inline void vec_dot_q3_K_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local float *x_df = (const local float *)x_qs + TXS_QS;
    const local int *x_sc = (const local int *)x_df + TXS_DM;
    const local int *y_qs = (const local int *)y + 4;
    const local float *y_df = (const local float *)y;

// unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K * VDR_Q3_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                const char *scales = 
                    ((const char *)(x_sc + i * (MMQ_TILE_NE_K / 8) + i / 8)) + k0 / 4;

                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q3_K_q8_1_impl_mmq(
                        &x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k0], 
                        &y_qs[j * MMQ_TILE_Y_K + k01], 
                        scales,
                        x_df[i], 
                        y_df[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
} 

)";

const char g_codeVecDotDp4a_Q4_K[] = R"(
inline void vec_dot_q4_K_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local half2 *x_dm = (const local half2 *)x_qs + TXS_QS;
    const local int *x_sc = (const local int *)x_dm + TXS_DM;
    const local int *y_qs = (const local int *)y + 4;
    const local half2 *y_ds = (const local half2 *)y;

// unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                const uchar *sc = 
                    (const uchar *)&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32] + 2 * (k01 / 16);

                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q4_K_q8_1_impl_mmq(
                        &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / 2], 
                        &y_qs[j * MMQ_TILE_Y_K + k01], 
                        sc, 
                        sc + 8,
                        x_dm[i], 
                        &y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

)";

const char g_codeVecDotDp4a_Q5_K[] = R"(
inline void vec_dot_q5_K_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local half2 *x_dm = (const local half2 *)x_qs + TXS_QS;
    const local int *x_sc = (const local int *)x_dm + TXS_DM;
    const local int *y_qs = (const local int *)y + 4;
    const local half2 *y_ds = (const local half2 *)y;

// unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR5_K * VDR_Q5_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                const uchar *sc = ((const uchar *)&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k00 / 32]) + 2 * (k01 / 16);

                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q5_K_q8_1_impl_mmq(
                        &x_qs[i * (QR5_K * MMQ_TILE_NE_K + 1) + k0], 
                        &y_qs[j * MMQ_TILE_Y_K + k01], 
                        sc, 
                        sc + 8,
                        x_dm[i], 
                        &y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
} 

)";

const char g_codeVecDotDp4a_Q6_K[] = R"(
inline void vec_dot_q6_K_q8_1_dp4a(
        const local int *x, 
        const local int *y, 
        float *sum, 
        const int k00) {

    const local int *x_qs = (const local int *)x;
    const local float *x_df = (const local float *)x_qs + TXS_QS;
    const local int *x_sc = (const local int *)x_df + TXS_DM;
    const local int *y_qs = (const local int *)y + 4;
    const local float *y_df = (const local float *)y;

// unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR6_K * VDR_Q6_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
            const int j = j0 + LID_1;

            unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
                const int i = i0 + LID_0;

                const char *sc = ((const char *)&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 16]);

                sum[j0 / NUM_SGS * MMQ_Y / SG_SIZE + i0 / SG_SIZE] += 
                    vec_dot_q6_K_q8_1_impl_mmq(
                        &x_qs[i * (QR6_K * MMQ_TILE_NE_K + 1) + k0], 
                        &y_qs[j * MMQ_TILE_Y_K + k01], 
                        sc,
                        x_df[i * (MMQ_TILE_NE_K / QI6_K) + i / QI6_K], 
                        &y_df[j * MMQ_TILE_Y_K + k01 / QI8_1]);
            }
        }
    }
}

)";

//
//    WriteBack
//

const char g_codeWriteBackDp4a[] = R"(
inline void mmq_write_back_dp4a(
        const float *sum, 
        const local int *ids_dst, 
        global float *dst,
        const int stride, 
        const int i_max, 
        const int j_max) {

    unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS) {
        const int j = j0 + LID_1;

        if (j > j_max) {
            return;
        }

        unroll_for (int i0 = 0; i0 < MMQ_Y; i0 += SG_SIZE) {
            const int i = i0 + LID_0;

            if (NEED_CHECK && i > i_max) {
                continue;
            }

            dst[ids_dst[j] * stride + i] = sum[(j0 / NUM_SGS) * (MMQ_Y / SG_SIZE) + i0 / SG_SIZE];
        }
    }
} 

)";

//
//    Kernel
//

const char g_kernelCodeMulMat[] = R"(
inline void mul_mat_q_process_tile(
        const global char *x, 
        const int offset_x, 
        const global int *y,
        const local int *ids_dst, 
        global float *dst, 
        global float *tmp_fixup,
        const int stride_row_x, 
        const int ncols_y, 
        const int stride_col_dst,
        const int tile_x_max_i, 
        const int tile_y_max_j, 
        const int kb0_start, 
        const int kb0_stop,
        local int *local_mem) {

    local int *tile_y = local_mem + NE_LOCAL_IDS;
    local int *tile_x = tile_y + NE_LOCAL_Y;

    float sum[MMQ_X * MMQ_Y / (NUM_SGS * SG_SIZE)] = {0.0f};

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += BLOCKS_PER_ITER) {
        LOAD_TILES(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const global int *by0 = y + ncols_y * (kb0 * QK / NE_BLOCK) * NE_BLOCK_Q8_1_MMQ;
            unroll_for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += NUM_SGS * SG_SIZE) {
                int l = l0 + LID_1 * SG_SIZE + LID_0;
                tile_y[l] = by0[l];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        VEC_DOT(tile_x, tile_y, sum, 0);

        barrier(CLK_LOCAL_MEM_FENCE);

        {
            const global int *by0 = y + ncols_y * ((kb0 * QK / NE_BLOCK) * NE_BLOCK_Q8_1_MMQ + NE_BLOCK_Q8_1_MMQ);
            unroll_for (int l0 = 0; l0 < MMQ_X * MMQ_TILE_Y_K; l0 += NUM_SGS * SG_SIZE) {
                int l = l0 + LID_1 * SG_SIZE + LID_0;
                tile_y[l] = by0[l];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        VEC_DOT(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (FIXUP) {
        WRITE_BACK(sum, ids_dst, tmp_fixup + GID_0 * (MMQ_X * MMQ_Y), MMQ_Y, MMQ_Y, MMQ_X);
    } else {
        WRITE_BACK(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
kernel void mul_mat_mm_q(
        const global char *x, 
        const global int *y, 
        const global int *ids_dst,
        const global int *expert_bounds, 
        global float *dst, 
        global float *tmp_fixup,
        const int ncols_x, 
        const int nrows_x, 
        const int ncols_dst, 
        const int stride_row_x, 
        const int ncols_y, 
        const int stride_col_dst,
        const int channel_ratio, 
        const int nchannels_y, 
        const int stride_channel_x, 
        const int stride_channel_y, 
        const int stride_channel_dst,
        const int sample_ratio, 
        const int nsamples_y, 
        const int stride_sample_x,  
        const int stride_sample_y, 
        const int stride_sample_dst,
        const int ncols_max
        SHAPE_INFO_ARGS) {

    x += X_BASE;
    y += Y_BASE;
    if (ids_dst != NULL) {
        ids_dst += IDS_DST_BASE;
    }
    if (expert_bounds != NULL) {
        expert_bounds += EXPERT_BOUNDS_BASE;
    }
    dst += DST_BASE;
    if (tmp_fixup != NULL) {
        tmp_fixup += TMP_FIXUP_BASE;
    }

    const int ntx = (ncols_max + MMQ_X - 1) / MMQ_X; // Number of tiles x
    const int nty = (nrows_x + MMQ_Y - 1) / MMQ_Y;   // Number of tiles y

    // divided into three sections: NE_LOCAL = NE_LOCAL_IDS + NE_LOCAL_Y + NE_LOCAL_X
    local int local_mem[NE_LOCAL];

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.

    // Stored at beginning of local memory.
    local int *ids_dst_shared = local_mem; 

    unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS * SG_SIZE) {
        const int j = j0 + LID_1 * SG_SIZE + LID_0;

        if (j0 + NUM_SGS * SG_SIZE > MMQ_X && j >= MMQ_X) {
            break;
        }

        ids_dst_shared[j] = j;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int wt = GID_2 / nchannels_y;
    const int zt = GID_2 - wt * nchannels_y;
    const int jt = GID_1;
    const int it = GID_0;

    // Defaults for regular matrix multiplication:
    int col_low = 0;
    int col_high = ncols_dst;
    int col_diff = ncols_dst;
    int offset_y = wt * stride_sample_y + zt * stride_channel_y;
    int offset_dst = wt * stride_sample_dst + zt * stride_channel_dst + jt * MMQ_X * stride_col_dst;

    if (ids_dst) {
        col_low = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y = 0;
        offset_dst = 0;

        if (jt * MMQ_X >= col_diff) {
            return;
        }

        // There is no previous tile that could cause a race condition
        // barrier(CLK_LOCAL_MEM_FENCE);

        unroll_for (int j0 = 0; j0 < MMQ_X; j0 += NUM_SGS * SG_SIZE) {
            const int j = j0 + LID_1 * SG_SIZE + LID_0;

            if (j0 + NUM_SGS * SG_SIZE > MMQ_X && j >= MMQ_X) {
                break;
            }

            ids_dst_shared[j] = ids_dst[col_low + jt * MMQ_X + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    offset_y += (col_low + jt * MMQ_X) * NE_BLOCK_Q8_1_MMQ;
    offset_dst += it * MMQ_Y;

    const int tile_x_max_i = nrows_x - it * MMQ_Y - 1;
    const int tile_y_max_j = col_diff - jt * MMQ_X - 1;

    const int offset_x = 
        (wt / sample_ratio) * stride_sample_x + 
        (zt / channel_ratio) * stride_channel_x + 
        it * MMQ_Y * stride_row_x;

    mul_mat_q_process_tile(
        x, 
        offset_x, 
        y + offset_y, 
        ids_dst_shared, 
        dst + offset_dst, 
        tmp_fixup, 
        stride_row_x, 
        ncols_y, 
        stride_col_dst,
        tile_x_max_i, 
        tile_y_max_j, 
        0, 
        ncols_x / QK,
        local_mem);
}

)";

} // namespace

//
//    LoadTiles
//

const char *MulMatQuantMmLoadTiles_Q4_0_Code() {
    return g_codeLoadTiles_Q4_0;
}

const char *MulMatQuantMmLoadTiles_Q4_1_Code() {
    return g_codeLoadTiles_Q4_1;
}

const char *MulMatQuantMmLoadTiles_Q5_0_Code() {
    return g_codeLoadTiles_Q5_0;
}

const char *MulMatQuantMmLoadTiles_Q5_1_Code() {
    return g_codeLoadTiles_Q5_1;
}

const char *MulMatQuantMmLoadTiles_Q8_0_Code() {
    return g_codeLoadTiles_Q8_0;
}

const char *MulMatQuantMmLoadTiles_Q2_K_Code() {
    return g_codeLoadTiles_Q2_K;
}

const char *MulMatQuantMmLoadTiles_Q3_K_Code() {
    return g_codeLoadTiles_Q3_K;
}

const char *MulMatQuantMmUnpackScales_Q45_K() {
    return g_coreUnpackScales_Q45_K;
}

const char *MulMatQuantMmLoadTiles_Q4_K_Code() {
    return g_codeLoadTiles_Q4_K;
}

const char *MulMatQuantMmLoadTiles_Q5_K_Code() {
    return g_codeLoadTiles_Q5_K;
}

const char *MulMatQuantMmLoadTiles_Q6_K_Code() {
    return g_codeLoadTiles_Q6_K;
}

const char *MulMatQuantMmLoadTiles_Mxfp4_Code() {
    return g_codeLoadTiles_Mxfp4;
}

//
//    VecDotDp4a
//

const char *VecDotDp4a_Q4_0_Code() {
    return g_codeVecDotDp4a_Q4_0;
}

const char *VecDotDp4a_Q4_1_Code() {
    return g_codeVecDotDp4a_Q4_1;
}

const char *VecDotDp4a_Q8_0_Code() {
    return g_codeVecDotDp4a_Q8_0;
}

const char *VecDotDp4a_Q8_1_Code() {
    return g_codeVecDotDp4a_Q8_1;
}

const char *VecDotDp4a_Q2_K_Code() {
    return g_codeVecDotDp4a_Q2_K;
}

const char *VecDotDp4a_Q3_K_Code() {
    return g_codeVecDotDp4a_Q3_K;
}

const char *VecDotDp4a_Q4_K_Code() {
    return g_codeVecDotDp4a_Q4_K;
}

const char *VecDotDp4a_Q5_K_Code() {
    return g_codeVecDotDp4a_Q5_K;
}

const char *VecDotDp4a_Q6_K_Code() {
    return g_codeVecDotDp4a_Q6_K;
}

//
//    WriteBack
//

const char *MulMatQuantMmWriteBackDp4aCode() {
    return g_codeWriteBackDp4a;
}

//
//    Kernel
//

const char *MulMatQuantMmKernelCode() {
    return g_kernelCodeMulMat;
}

} // namespace kernels
} // namespace onednn
} // namespace arhat

