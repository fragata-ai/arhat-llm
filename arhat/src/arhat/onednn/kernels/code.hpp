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

#pragma once

namespace arhat {
namespace onednn {
namespace kernels {

const char *AddIdSimpleKernelCode();
const char *ArgSortSimpleKernelCode();
const char *BinarySimpleAddOpCode();
const char *BinarySimpleSubOpCode();
const char *BinarySimpleMulOpCode();
const char *BinarySimpleDivOpCode();
const char *BinarySimpleKernelCode();
const char *BinarySimpleUnravelKernelCode();
const char *CpySimpleKernelCode();
const char *CumSumSimpleBlkKernelCode();
const char *CumSumSimpleAddKernelCode();
const char *DiagSimpleKernelCode();
const char *FattnSimpleKernelCode();
const char *FattnSimpleQ1KernelCode();
const char *FattnTileKernelCode();
const char *FattnVecKernelCode();
const char *FillSimpleKernelCode();
const char *GdnSimpleKernelCode();
const char *GetRowsSimpleKernelCode();
const char *GluSimpleSwigluOaiOpCode();
const char *GluSimpleKernelCode();
const char *GluSimpleSwigluOaiKernelCode();
const char *GroupNormSimpleKernelCode();
const char *L2NormSimpleKernelCode();
const char *MulMatMmKernelCode();
const char *MulMatIdMmKernelCode();
const char *MulMatQuantMmKernelCode();
const char *MulMatIdHelperKernelCode();
const char *MulMatIdQuantSimple_Q4_0_KernelCode();
const char *MulMatIdQuantSimple_Q4_K_KernelCode();
const char *MulMatIdQuantSimple_Q5_0_KernelCode();
const char *MulMatIdQuantSimple_Q6_K_KernelCode();
const char *MulMatIdQuantSimple_Q8_0_KernelCode();
const char *MulMatIdQuantSimple_Mxfp4_KernelCode();
const char *MulMatQuantSimple_Q4_0_KernelCode();
const char *MulMatQuantSimple_Q4_K_KernelCode();
const char *MulMatQuantSimple_Q5_0_KernelCode();
const char *MulMatQuantSimple_Q6_K_KernelCode();
const char *MulMatQuantSimple_Q8_0_KernelCode();
const char *MulMatQuantSimple_Mxfp4_KernelCode();
const char *MulMatQuantVecKernelCode();
const char *MulMatQuantVecV2KernelCode();
const char *MulMatQuantVecV2OptKernelCode();
const char *MulMatVecKernelCode();
const char *MulMatVecV2KernelCode();
const char *MulMatVecV2_Q2_K_KernelCode();
const char *MulMatVecV2_Q3_K_KernelCode();
const char *MulMatVecV2_Q4_K_KernelCode();
const char *MulMatVecV2_Q5_K_KernelCode();
const char *MulMatVecV2_Q6_K_KernelCode();
const char *NormSimpleKernelCode();
const char *PadSimpleKernelCode();
const char *PadSimpleCircularKernelCode();
const char *QuantizeMm_Mfxp4_KernelCode();
const char *QuantizeMm_Q8_1_KernelCode();
const char *QuantizeVec_Q8_1_KernelCode();
const char *QuantizeVec_Q8_1_Soa_KernelCode();
const char *QuantizeV2_Q8_1_X4_KernelCode();
const char *RepeatSimpleKernelCode();
const char *RmsNormSimpleKernelCode();
const char *RopeSimpleCommonCode();
const char *RopeSimpleNormKernelCode();
const char *RopeSimpleNeoxKernelCode();
const char *RopeSimpleMultiKernelCode();
const char *RopeSimpleVisionKernelCode();
const char *SetSimpleKernelCode();
const char *SetRowsSimpleKernelCode();
const char *SolveTriSimpleKernelCode();
const char *SsmConvSimpleKernelCode();
const char *SsmConvSimpleX4KernelCode();
const char *TriSimpleKernelCode();
const char *UnarySimpleSqrOpCode();
const char *UnarySimpleSqrtOpCode();
const char *UnarySimpleLogOpCode();
const char *UnarySimpleSinOpCode();
const char *UnarySimpleCosOpCode();
const char *UnarySimpleAbsOpCode();
const char *UnarySimpleSgnOpCode();
const char *UnarySimpleNegOpCode();
const char *UnarySimpleStepOpCode();
const char *UnarySimpleTanhOpCode();
const char *UnarySimpleEluOpCode();
const char *UnarySimpleReluOpCode();
const char *UnarySimpleSigmoidOpCode();
const char *UnarySimpleGeluOpCode();
const char *UnarySimpleGeluQuickOpCode();
const char *UnarySimpleSiluOpCode();
const char *UnarySimpleHardswishOpCode();
const char *UnarySimpleHardsigmoidOpCode();
const char *UnarySimpleExpOpCode();
const char *UnarySimpleExpm1OpCode();
const char *UnarySimpleSoftplusOpCode();
const char *UnarySimpleGeluErfOpCode();
const char *UnarySimpleFloorOpCode();
const char *UnarySimpleCeilOpCode();
const char *UnarySimpleRoundOpCode();
const char *UnarySimpleTruncOpCode();
const char *UnarySimpleKernelCode();
const char *UnarySimpleXieluKernelCode();

const char *MulMatMmaF32Code();

const char *MulMatQuantMmLoadTiles_Q4_0_Code();
const char *MulMatQuantMmLoadTiles_Q4_1_Code();
const char *MulMatQuantMmLoadTiles_Q5_0_Code();
const char *MulMatQuantMmLoadTiles_Q5_1_Code();
const char *MulMatQuantMmLoadTiles_Q8_0_Code();
const char *MulMatQuantMmLoadTiles_Q2_K_Code();
const char *MulMatQuantMmLoadTiles_Q3_K_Code();
const char *MulMatQuantMmUnpackScales_Q45_K();
const char *MulMatQuantMmLoadTiles_Q4_K_Code();
const char *MulMatQuantMmLoadTiles_Q5_K_Code();
const char *MulMatQuantMmLoadTiles_Q6_K_Code();
const char *MulMatQuantMmLoadTiles_Mxfp4_Code();

const char *MulMatQuantMmWriteBackDp4aCode();

const char *VecDotDp4a_Q4_0_Code();
const char *VecDotDp4a_Q4_1_Code();
const char *VecDotDp4a_Q8_0_Code();
const char *VecDotDp4a_Q8_1_Code();
const char *VecDotDp4a_Q2_K_Code();
const char *VecDotDp4a_Q3_K_Code();
const char *VecDotDp4a_Q4_K_Code();
const char *VecDotDp4a_Q5_K_Code();
const char *VecDotDp4a_Q6_K_Code();

const char *VecDotQuantCommonCode();

const char *VecDotDefs_Q4_0_Code();
const char *VecDotDefs_Q4_1_Code();
const char *VecDotDefs_Q5_0_Code();
const char *VecDotDefs_Q5_1_Code();
const char *VecDotDefs_Q8_0_Code();
const char *VecDotDefs_Q2_K_Code();
const char *VecDotDefs_Q3_K_Code();
const char *VecDotDefs_Q4_K_Code();
const char *VecDotDefs_Q5_K_Code();
const char *VecDotDefs_Q6_K_Code();
const char *VecDotDefs_Mxfp4_Code();

const char *VecDot_Q4_0_Code();
const char *VecDot_Q4_1_Code();
const char *VecDot_Q5_0_Code();
const char *VecDot_Q5_1_Code();
const char *VecDot_Q8_0_Code();
const char *VecDot_Q2_K_Code();
const char *VecDot_Q3_K_Code();
const char *VecDot_Q4_K_Code();
const char *VecDot_Q5_K_Code();
const char *VecDot_Q6_K_Code();
const char *VecDot_Mxfp4_Code();

const char *VecDotImpl_Q4_0_Code();
const char *VecDotImpl_Q4_1_Code();
const char *VecDotImpl_Q5_0_Code();
const char *VecDotImpl_Q5_1_Code();
const char *VecDotImpl_Q8_0_Code();
const char *VecDotImpl_Q8_1_Code();
const char *VecDotImpl_Q2_K_Code();
const char *VecDotImpl_Q3_K_Code();
const char *VecDotImpl_Q4_K_Code();
const char *VecDotImpl_Q5_K_Code();
const char *VecDotImpl_Q6_K_Code();

const char *VecDotMmImpl_Q2_K_Code();
const char *VecDotMmImpl_Q3_K_Code();
const char *VecDotMmImpl_Q4_K_Code();
const char *VecDotMmImpl_Q5_K_Code();
const char *VecDotMmImpl_Q6_K_Code();

const char *MulMatVecV2CommonCode();

const char *MulMatVecV2Impl_Q4_0_Code();
const char *MulMatVecV2Impl_Q4_1_Code();
const char *MulMatVecV2Impl_Q5_0_Code();
const char *MulMatVecV2Impl_Q5_1_Code();
const char *MulMatVecV2Impl_Q8_0_Code();
const char *MulMatVecV2Impl_Q2_K_Code();
const char *MulMatVecV2Impl_Q3_K_Code();
const char *MulMatVecV2Impl_Q4_K_Code();
const char *MulMatVecV2Impl_Q5_K_Code();
const char *MulMatVecV2Impl_Q6_K_Code();
const char *MulMatVecV2Impl_Mxfp4_Code();

const char *MulMatQuantVecV2BaseCode();

const char *MulMatQuantVecV2Defs_Q4_0_Code();
const char *MulMatQuantVecV2Defs_Q4_1_Code();
const char *MulMatQuantVecV2Defs_Q5_0_Code();
const char *MulMatQuantVecV2Defs_Q5_1_Code();
const char *MulMatQuantVecV2Defs_Q8_0_Code();
const char *MulMatQuantVecV2Defs_Q8_1_Code();
const char *MulMatQuantVecV2Defs_Q2_K_Code();
const char *MulMatQuantVecV2Defs_Q3_K_Code();
const char *MulMatQuantVecV2Defs_Q4_K_Code();
const char *MulMatQuantVecV2Defs_Q5_K_Code();
const char *MulMatQuantVecV2Defs_Q6_K_Code();
const char *MulMatQuantVecV2Defs_Mxfp4_Code();

const char *MulMatQuantVecV2Impl_Q4_0_Code();
const char *MulMatQuantVecV2Impl_Q4_1_Code();
const char *MulMatQuantVecV2Impl_Q5_0_Code();
const char *MulMatQuantVecV2Impl_Q5_1_Code();
const char *MulMatQuantVecV2Impl_Q8_0_Code();
const char *MulMatQuantVecV2ImplLegacyCode();
const char *MulMatQuantVecV2Impl_Q2_K_Code();
const char *MulMatQuantVecV2Impl_Q3_K_Code();
const char *MulMatQuantVecV2Impl_Q4_K_Code();
const char *MulMatQuantVecV2Impl_Q5_K_Code();
const char *MulMatQuantVecV2Impl_Q45_K_Code();
const char *MulMatQuantVecV2Impl_Q6_K_Code();
const char *MulMatQuantVecV2Impl_Mxfp4_Code();

const char *MulMatQuantVecV2OptImpl_Q4_0_Code();
const char *MulMatQuantVecV2OptImpl_Q4_1_Code();
const char *MulMatQuantVecV2OptImpl_Q5_0_Code();
const char *MulMatQuantVecV2OptImpl_Q5_1_Code();
const char *MulMatQuantVecV2OptImpl_Q8_0_Code();
const char *MulMatQuantVecV2OptImpl_Q2_K_Code();
const char *MulMatQuantVecV2OptImpl_Q3_K_Code();
const char *MulMatQuantVecV2OptImpl_Q4_K_Code();
const char *MulMatQuantVecV2OptImpl_Q5_K_Code();
const char *MulMatQuantVecV2OptImpl_Q45_K_Code();
const char *MulMatQuantVecV2OptImpl_Q6_K_Code();
const char *MulMatQuantVecV2OptImpl_Mxfp4_Code();

} // namespace kernels
} // namespace onednn
} // namespace arhat

