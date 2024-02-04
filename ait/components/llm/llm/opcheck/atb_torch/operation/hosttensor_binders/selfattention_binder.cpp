/**
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
#include "selfattention_binder.h"

SelfAttentionBinder::SelfAttentionBinder() {}

SelfAttentionBinder::~SelfAttentionBinder() {}

void SelfAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    if (paramJson.contains("token_offset")) {
        for (auto &item : paramJson["token_offset"]) {
            tokenOffset_.push_back(item.get<int32_t>();
        }
    }
    seqLen_.clear();
    for (auto item : paramJson["seq_len"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
}

void SelfAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    if (variantPack.inTensors.size() == 5) { // 5: flash encoder input num
        const uint32_t seqLenTensorId = 4;
        variantPack.inTensors.at(seqLenTensorId).hostData = seqLen.data();
    } else {
        const uint32_t tokenOffsetTensorId = 6; // 6: 设置tokenOffset的tensor位置
        const uint32_t seqLenTensorId = 7; // 7: 设置seqLen的tensor位置
        variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset.data();
        variantPack.inTensors.at(seqLenTensorId).hostData = seqLen.data();
    }
}