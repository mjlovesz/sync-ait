/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_TRAINOPPARAM_H
#define ATB_TRAINOPPARAM_H
#include <cstdint>
#include <string>
#include <limits>
#include <hccl/hccl_types.h>
#include <acl/acl.h>
#include "atb/svector.h"

namespace atb {
namespace train {
struct GenAttentionMaskParam {
    int32_t headNum = 1;
    atb::SVector<int32_t> seqLen;
};

struct RopeGradParam {
    std::vector<int32_t> qSeqLen;
};

struct FastSoftMaxParam {
    int32_t headNum = 0;
    std::vector<int32_t> qSeqLen;
};

struct FastSoftMaxGradParam {
    int32_t headNum = 0;
    std::vector<int32_t> qSeqLen;
};

struct StridedBatchMatmulParam {
    bool transposeA = false;
    bool transposeB = false;
    int32_t batch = 1;
    int32_t headNum = 1;
    std::vector<int32_t> m;
    std::vector<int32_t> n;
    std::vector<int32_t> k;
    std::vector<int32_t> lda;
    std::vector<int32_t> ldb;
    std::vector<int32_t> ldc;
    std::vector<int32_t> strideA;
    std::vector<int32_t> strideB;
    std::vector<int32_t> strideC;
};

struct FlashAttentionBackwardParam {
    float scaleValue = 0;
    int64_t headNum = 1;
    float keepProb = 1.0;
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 1;
    int64_t preciseMode = 0;
    int64_t groups = -1;
    enum IoLayout : int {
        BNSD = 0,
        BSH,
        SBH
    };
    IoLayout ioLayout = BNSD;
};

struct FlashAttentionParam {
    float scaleValue = 0;
    int64_t headNum = 1;
    float keepProb = 1.0;
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 1;
    int64_t preciseMode = 0;
    int64_t groups = -1;
    enum IoLayout : int {
        BNSD = 0,
        BSH,
        SBH
    };
    IoLayout ioLayout = BNSD;
};

struct UnpadWithHiddenStateParam {
    std::vector<int32_t> qSeqLen;
    int32_t maxSeqLen;
};

struct PadWithHiddenStateParam {
    std::vector<int32_t> qSeqLen;
    int32_t maxSeqLen;
};

struct RmsNormBackwardParam {};

} // namespace train
} // namespace atb_train
#endif