/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/ 

// Based on the code from oneDNN 3.10
// (https://github.com/uxlfoundation/oneDNN)
// modified by FRAGATA COMPUTER SYSTEMS AG

#include <cstdint>
#include <cassert>

#include "arhat/core/runtime.hpp"

namespace arhat {
namespace core {

namespace {

int64_t Volume(const int64_t *ne) {
    int64_t prod = 1;
    for (int i = 0; i < MaxDims; i++) {
        prod *= ne[i];
    }
    return prod;
}

enum class ReshapeAction {
    REMOVE,
    ADD,
    KEEP,
    REARRANGE,
    FAIL
};

ReshapeAction FindReshapeGroups(
        const int64_t *iDims,
        const int64_t *oDims,
        int &iGroupBegin,
        int iGroupEnd,
        int &oGroupBegin,
        int oGroupEnd) {
    // 1st step: check for `1` in the input dims
    if (iGroupEnd > 0 && iDims[iGroupEnd - 1] == 1) {
        iGroupBegin = iGroupEnd - 1;
        oGroupBegin = oGroupEnd;
        return ReshapeAction::REMOVE;
    } 

    // 2nd step: check for `1` in the output dims
    if (oGroupEnd > 0 && oDims[oGroupEnd - 1] == 1) {
        iGroupBegin = iGroupEnd;
        oGroupBegin = oGroupEnd - 1;
        return ReshapeAction::ADD;
    }

    // at this moment both groups cannot be empty
    if (iGroupEnd == 0 || oGroupEnd == 0) {
        return ReshapeAction::FAIL;
    }

    // 3rd step: find the non-trivial groups of the same volume
    iGroupBegin = iGroupEnd - 1;
    oGroupBegin = oGroupEnd - 1;

    int64_t iVolume = iDims[iGroupBegin];
    int64_t oVolume = oDims[oGroupBegin];

    while (iVolume != oVolume) {
        if (iVolume < oVolume) {
            if (iGroupBegin == 0) {
                return ReshapeAction::FAIL;
            }
            iGroupBegin--;
            iVolume *= iDims[iGroupBegin];
        } else {
            if (oGroupBegin == 0) {
                return ReshapeAction::FAIL;
            }
            oGroupBegin--;
            oVolume *= oDims[oGroupBegin];
        }
    }
  
    assert(iVolume == oVolume);
    assert(iGroupBegin >= 0);
    assert(oGroupBegin >= 0);

    if (iGroupBegin + 1 == iGroupEnd && oGroupBegin + 1 == oGroupEnd) {
        return ReshapeAction::KEEP;
    } else {
        return ReshapeAction::REARRANGE; 
    }
}

} // namespace

//
//    Public functions
//

// Algorithm borrowed from oneDNN [common/memory_desc.cpp]
bool CanReshape(
        const int64_t *srcNe,
        const size_t *srcNb,
        const int64_t *dstNe,
        const size_t *dstNb) {
    int64_t iDims[MaxDims];
    int64_t iStrides[MaxDims];
    int64_t oDims[MaxDims];
    int64_t oStrides[MaxDims];
    int64_t tStrides[MaxDims];
    for (int d = 0; d < MaxDims; d++) {
        iDims[d] = srcNe[MaxDims - 1 - d];
        iStrides[d] = int64_t(srcNb[MaxDims - 1 - d]);
        oDims[d] = srcNe[MaxDims - 1 - d];
        oStrides[d] = int64_t(srcNb[MaxDims - 1 - d]);
        tStrides[d] = 0;
    }
    int iGroupBegin = MaxDims;
    int iGroupEnd = MaxDims;
    int oGroupBegin = MaxDims;
    int oGroupEnd = MaxDims;

    while (iGroupEnd != 0 || oGroupEnd != 0) {
        ReshapeAction action = 
            FindReshapeGroups(
                iDims,
                oDims,
                iGroupBegin, 
                iGroupEnd, 
                oGroupBegin, 
                oGroupEnd);
        if (action == ReshapeAction::REMOVE) {
            // nothing to do
        } else if (action == ReshapeAction::ADD) {
            // get the stride from the right 
            int64_t currStride = 
                (iGroupBegin == MaxDims) ? 
                    1 : 
                    iStrides[iGroupBegin] * iDims[iGroupBegin];
            tStrides[oGroupBegin] = currStride;
        } else if (action == ReshapeAction::KEEP) {
            // change the axis index from 'iGroupBegin' to 'oGroupBegin'
            assert(iGroupBegin + 1 == iGroupEnd);
            assert(oGroupBegin + 1 == oGroupEnd); 
            tStrides[oGroupBegin] = iStrides[iGroupBegin];
        } else if (action == ReshapeAction::REARRANGE) {
            // check that input group is dense, sequential, and is not blocked
            for (int d = iGroupEnd - 1; d > iGroupBegin; d--) {
                if (iDims[d] * iStrides[d] != iStrides[d - 1]) {
                    return false;
                }
            }
            int64_t currStride = iStrides[iGroupEnd - 1];
            for (int d = oGroupEnd - 1; d >= oGroupBegin; d--) {
                tStrides[d] = currStride;
                currStride *= oDims[d];
            } 
        } else {
            assert(action == ReshapeAction::FAIL);
            return false;
        }
        iGroupEnd = iGroupBegin;
        oGroupEnd = oGroupBegin;
    }

    for (int d = 0; d < MaxDims; d++) {
        if (tStrides[d] != oStrides[d]) {
            return false;
        }
    }
    return true;
}

} // namespace core
} // namespace arhat

