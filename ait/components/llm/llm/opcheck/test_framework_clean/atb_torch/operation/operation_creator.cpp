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
#include "operation_creator.h"
#include <nlohmann/json.hpp>
#include <functional>
#include "atb/log.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"

using OperationCreateFunc = std::function<atb::Status(const nlohmann::json &paramJson, atb::Operation **op)>;

static atb::Status AllReduceOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AllReduceParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("allReduceType") != paramJson.end()) {
        param.allReduceType = paramJson["allReduceType"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    ATB_LOG(INFO) << "AllReduceParam rank:" << param.rank;
    ATB_LOG(INFO) << "AllReduceParam rankSize:" << param.rankSize;
    return CreateOperation(param, op);
}

static atb::Status BroadcastOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::BroadcastParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    ATB_LOG(INFO) << "BroadcastParam rank:" << param.rank << "rankSize:" << param.rankSize;
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    ATB_LOG(INFO) << "BroadcastParam rank:" << param.rank << "rankSize:" << param.rankSize << "rankRoot:" <<
        param.rankRoot << "backend:" << param.backend;
    return CreateOperation(param, op);
}

static atb::Status AllGatherOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AllGatherParam param;
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    ATB_LOG(INFO) << "AllGatherParam rank:" << param.rank;
    ATB_LOG(INFO) << "AllGatherParam rankSize:" << param.rankSize;
    ATB_LOG(INFO) << "AllGatherParam backend:" << param.backend;
    return CreateOperation(param, op);
}

static atb::Status LinearParallelOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearParallelParam param;
    if (paramJson.find("transWeight") != paramJson.end()) {
        param.transWeight = paramJson["transWeight"].get<bool>();
    }
    if (paramJson.find("bias") != paramJson.end()) {
        param.bias = paramJson["bias"].get<std::string>();
    }
    if (paramJson.find("rankRoot") != paramJson.end()) {
        param.rankRoot = paramJson["rankRoot"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.contains("commMode")) {
        param.commMode = atb::infer::CommMode(paramJson["commMode"].get<int>());
    }
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    param.parallelType = paramJson["parallelType"].get<std::string>();
    return CreateOperation(param, op);
}

static atb::Status CumsumOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::CumsumParam param;
    for (auto item : paramJson["axes"]) {
        param.axes.push_back(item.get<int64_t>());
        ATB_LOG(FATAL) << "axes:" << param.axes.at(0);
    }
    if (paramJson.contains("exclusive")) {
        param.exclusive = paramJson["exclusive"].get<bool>();
    }
    if (paramJson.contains("reverse")) {
        param.reverse = paramJson["reverse"].get<bool>();
    }
    return CreateOperation(param, op);
}

static atb::Status ConcatOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ConcatParam param;
    ATB_LOG(INFO) << "ConcatParam axis:" << param.concatDim;
    if (paramJson.contains("concatDim")) {
        param.concatDim = paramJson["concatDim"].get<int>();
    } 
    return CreateOperation(param, op);
}

static atb::Status SplitOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SplitParam param;
    if (paramJson.contains("splitDim")) {
        param.splitDim = paramJson["splitDim"].get<int>();
    }
    if (paramJson.contains("splitNum")) {
        param.splitNum = paramJson["splitNum"].get<int>();
    }
    return CreateOperation(param, op);
}

static atb::Status GatherOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::GatherParam param;
    ATB_LOG(INFO) << "GatherParam axis:" << param.axis;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int64_t>();
    }
    return CreateOperation(param, op);
}

static atb::Status LinearOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    ATB_LOG(INFO) << "LinearParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias;
    return CreateOperation(param, op);
}

static atb::Status MatmulOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::MatmulParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    ATB_LOG(INFO) << "MatmulParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB;
    return CreateOperation(param, op);
}

static atb::Status SelfAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SelfAttentionParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.contains("qScale")) {
        param.qScale = paramJson["qScale"].get<float>();
    }
    if (paramJson.contains("qkScale")) {
        param.qkScale = paramJson["qkScale"].get<float>();
    }
    if (paramJson.contains("headDim")) {
        param.headDim = paramJson["headDim"].get<int>();
    }
    if (paramJson.contains("isSupportAlibi")) {
        param.isSupportAlibi = paramJson["isSupportAlibi"].get<bool>();
    }
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }
    if (paramJson.contains("isEncoder")) {
        param.isEncoder = paramJson["isEncoder"].get<bool>();
    }
    if (paramJson.contains("isClamp")) {
        param.isClamp = paramJson["isClamp"].get<bool>();
    }
    if (paramJson.contains("clampMin")) {
        param.clampMin = paramJson["clampMin"].get<float>();
    }
    if (paramJson.contains("clampMax")) {
        param.clampMax = paramJson["clampMax"].get<float>();
    }
    ATB_LOG(INFO) << "SelfAttentionParam headNum:" << param.headNum << ", qScale:" << param.qScale
                  << ", headDim:" << param.headDim << ", qkScale:" << param.qkScale
                  << ", isSupportAlibi:" << param.isSupportAlibi 
                  << ", kvHeadNum:" << param.kvHeadNum << ", isEncoder:" << param.isEncoder;
    return CreateOperation(param, op);
}

static atb::Status SetValueOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SetValueParam param;
    for (auto item : paramJson["starts"]) {
        param.starts.push_back(item.get<int>());
        ATB_LOG(INFO) << "starts:" << param.starts.at(0);
    }
    for (auto item : paramJson["ends"]) {
        param.ends.push_back(item.get<int>());
        ATB_LOG(INFO) << "ends:" << param.ends.at(0);
    }
    for (auto item : paramJson["strides"]) {
        param.strides.push_back(item.get<int>());
        ATB_LOG(INFO) << "strides:" << param.strides.at(0);
    }
    return CreateOperation(param, op);
}

static atb::Status SortOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SortParam param;
    for (auto item : paramJson["num"]) {
        param.num.push_back(item.get<int>());
        ATB_LOG(INFO) << "num:" << param.num.at(0);
    }
    return CreateOperation(param, op);
}

static atb::Status TransposeOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::TransposeParam param;
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "transpose(" << param.perm << ")";
    return CreateOperation(param, op);
}

static atb::Status LinearActivationQuantOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearActivationQuantParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("activationFuncType")) {
        param.activationFuncType = atb::infer::ActivationType(paramJson["activationFuncType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "LinearActivationQuantParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB 
    << ", hasBias:" << param.hasBias << ", activationFuncType:" << param.activationFuncType;

    return CreateOperation(param, op);
}

static atb::Status LinearSparseOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearSparseParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("tilingK")) {
        param.tilingK = paramJson["tilingK"].get<uint32_t>();
    }
    if (paramJson.contains("tilingN")) {
        param.tilingN = paramJson["tilingN"].get<uint32_t>();
    }
    ATB_LOG(INFO) << "LinearSparseParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", tilingK:" << param.tilingK << ", tilingN:" << param.tilingN;

    return CreateOperation(param, op);
}

static atb::Status ActivationOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ActivationParam param;
    if (paramJson.contains("activationType")) {
        param.activationType = atb::infer::ActivationType(paramJson["activationType"].get<int32_t>());
    }
    if (paramJson.contains("scale")) {
        param.scale = paramJson["scale"].get<float>();
    }
    ATB_LOG(INFO) << "ActivationParam activationType:" << param.activationType << ", scale:" << param.scale;
    return CreateOperation(param, op);
}

static atb::Status RopeOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RopeParam param;
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
    if (paramJson.contains("cosFormat")) {
        param.cosFormat = paramJson["cosFormat"].get<int>();
    }
    return CreateOperation(param, op);
}

static atb::Status KvCacheOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::KvCacheParam param;
    return CreateOperation(param, op);
}

static atb::Status PagedAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::PagedAttentionParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.qkScale = paramJson["qkScale"].get<float>();
    param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    if (paramJson.contains("isSupportAlibi")) {
        param.isSupportAlibi = paramJson["isSupportAlibi"].get<bool>();
    }

    if (paramJson.contains("maskType")) {
        param.maskType = atb::infer::PagedAttentionParam::MaskType(paramJson["maskType"].get<int32_t>());
    }

    if (paramJson.contains("quantType")) {
        param.quantType = atb::infer::PagedAttentionParam::QuantType(paramJson["quantType"].get<int32_t>());
    }

    if (paramJson.contains("hasQuantOffset")) {
        param.hasQuantOffset = paramJson["hasQuantOffset"].get<bool>();
    }
    ATB_LOG(INFO) << "PagedAttentionOperationCreate headNum:" << param.headNum << ", scale:" << param.qkScale 
                  << ", kvHeadNum:" << param.kvHeadNum << ", isSupportAlibi:" << param.isSupportAlibi 
                  << ", quantType:" << param.quantType << ", hasQuantOffset:" << param.hasQuantOffset;
    return CreateOperation(param, op);
}

static atb::Status ReshapeAndCacheOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ReshapeAndCacheParam param;
    return CreateOperation(param, op);
}

static atb::Status LinearQuantOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearQuantParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    ATB_LOG(INFO) << "LinearQuantParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB 
                  << ", hasBias:" << param.hasBias;
    return CreateOperation(param, op);
}

static atb::Status LinearActivationOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LinearActivationParam param;
    if (paramJson.contains("transposeA")) {
        param.transposeA = paramJson["transposeA"].get<bool>();
    }
    if (paramJson.contains("transposeB")) {
        param.transposeB = paramJson["transposeB"].get<bool>();
    }
    if (paramJson.contains("hasBias")) {
        param.hasBias = paramJson["hasBias"].get<bool>();
    }
    if (paramJson.contains("activationFuncType")) {
        param.activationFuncType = atb::infer::ActivationType(paramJson["activationFuncType"].get<int32_t>());
    }
    ATB_LOG(INFO) << "LinearActivationParam transposeA:" << param.transposeA << ", transposeB:" << param.transposeB
                  << ", hasBias:" << param.hasBias << ", activationFuncType:" << param.activationFuncType;
    return CreateOperation(param, op);
}

static atb::Status FillOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::FillParam param;
    if (paramJson.contains("withMask")) {
        param.withMask = paramJson["withMask"].get<bool>();
    }
    if (paramJson.contains("value")) {
        for (auto item : paramJson["value"]) {
            param.value.push_back(item.get<float>());
        }
    }
    if (paramJson.contains("outDim")) {
        for (auto item : paramJson["outDim"]) {
            param.outDim.push_back(item.get<int32_t>());
        }
    }
    ATB_LOG(INFO) << "FillParam withMask:" << param.withMask << ", value:" << param.value
                  << ", outDim:" << param.outDim;   
    return CreateOperation(param, op);
}

static atb::Status RepeatOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RepeatParam param;
    for (auto item : paramJson["multiples"]) {
        param.multiples.push_back(item.get<int64_t>());
    }
    ATB_LOG(INFO) << "RepeatParam multiples:" << param.multiples;
    return CreateOperation(param, op);
}

static atb::Status SliceOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SliceParam param;
    for (auto item : paramJson["offsets"]) {
        param.offsets.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["size"]) {
        param.size.push_back(item.get<int64_t>());
    }
    return CreateOperation(param, op);
}   

static atb::Status SoftmaxOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::SoftmaxParam param;
    for (auto item : paramJson["axes"]) {
        param.axes.push_back(item.get<int64_t>());
    }
    return CreateOperation(param, op);
}

static atb::Status ElewiseOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ElewiseParam param;
    param.elewiseType = paramJson["elewiseType"].get<atb::infer::ElewiseParam::ElewiseType>();
    if (paramJson.contains("varAttr")) {
        param.mulsParam.varAttr = paramJson["varAttr"].get<float>();
    }
    if (paramJson.contains("outTensorType")) {
        param.outTensorType = paramJson["outTensorType"].get<aclDataType>();
    }
    if (paramJson.contains("inputScale")) {
        param.quantParam.inputScale = paramJson["inputScale"].get<float>();
    }
    if (paramJson.contains("inputOffset")) {
        param.quantParam.inputOffset = paramJson["inputOffset"].get<int>();
    }
    return CreateOperation(param, op);
}

static atb::Status TopkToppSamplingOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::TopkToppSamplingParam param;
    param.randSeed = paramJson["randSeed"].get<uint32_t>();
    param.topk = paramJson["topk"].get<uint32_t>();
    return CreateOperation(param, op);
}

static atb::Status PadOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::PadParam param;
    return CreateOperation(param, op);
}

static atb::Status UnpadOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::UnpadParam param;
    return CreateOperation(param, op);
}

static atb::Status GenAttentionMaskOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::GenAttentionMaskParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
        ATB_LOG(INFO) << "param.headNum:" << param.headNum;
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "param.seqLen:" << param.seqLen;
    return CreateOperation(param, op);
}

static atb::Status RopeGradOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::RopeGradParam param;
    for (auto item : paramJson["qSeqLen"]) {
        param.qSeqLen.push_back(item.get<int>());
    }
    return CreateOperation(param, op);
}

static atb::Status RmsNormBackwardOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::RmsNormBackwardParam param;
    return CreateOperation(param, op);
}

static atb::Status LayerNormOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::LayerNormParam param;
    if (paramJson.contains("layerType")) {
        param.layerType = atb::infer::LayerNormParam::LayerNormType(paramJson["layerType"].get<int32_t>());
    }
    if (param.layerType == atb::infer::LayerNormParam::LAYER_NORM_NORM) {
        if (paramJson.contains("epsilon")) {
            param.normParam.epsilon = paramJson["epsilon"].get<float>();
        }
        if (paramJson.contains("quantType")) {
            param.normParam.quantType = atb::infer::QuantType(paramJson["quantType"].get<int32_t>());
        }
        if (paramJson.contains("beginNormAxis")) {
            param.normParam.beginNormAxis = paramJson["beginNormAxis"].get<int32_t>();
        }
        if (paramJson.contains("beginParamsAxis")) {
            param.normParam.beginParamsAxis = paramJson["beginParamsAxis"].get<int32_t>();
        }
        if (paramJson.contains("quantInputScale")) {
            param.normParam.quantInputScale = paramJson["quantInputScale"].get<float>();
        }
        if (paramJson.contains("quantInputOffset")) {
            param.normParam.quantInputOffset = paramJson["quantInputOffset"].get<int>();
        }
        if (paramJson.contains("quantInputAlpha")) {
            param.normParam.quantInputAlpha = paramJson["quantInputAlpha"].get<float>();
        }
    }
    if (param.layerType == atb::infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        if (paramJson.contains("epsilon")) {
            param.postNormParam.epsilon = paramJson["epsilon"].get<float>();
        }
        if (paramJson.contains("quantType")) {
            param.postNormParam.quantType = atb::infer::QuantType(paramJson["quantType"].get<int32_t>());
        }
        if (paramJson.contains("opMode")) {
            param.postNormParam.opMode = paramJson["opMode"].get<size_t>();
        }
        if (paramJson.contains("zoomScaleValue")) {
            param.postNormParam.zoomScaleValue = paramJson["zoomScaleValue"].get<float>();
        }
        if (paramJson.contains("quantInputScale")) {
            param.postNormParam.quantInputScale = paramJson["quantInputScale"].get<float>();
        }
        if (paramJson.contains("quantInputOffset")) {
            param.postNormParam.quantInputOffset = paramJson["quantInputOffset"].get<int>();
        }
        if (paramJson.contains("quantInputAlpha")) {
            param.postNormParam.quantInputAlpha = paramJson["quantInputAlpha"].get<float>();
        }
    }
    return CreateOperation(param, op);
}

static atb::Status RmsNormOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::RmsNormParam param;
    if (paramJson.contains("layerType")) {
        param.layerType = atb::infer::RmsNormParam::RmsNormType(paramJson["layerType"].get<int32_t>());
    }
    if (paramJson.contains("rstd")) {
        param.normParam.rstd = paramJson["rstd"].get<bool>();
    }
    if (param.layerType == atb::infer::RmsNormParam::RMS_NORM_NORM) {
        if (paramJson.contains("epsilon")) {
            param.normParam.epsilon = paramJson["epsilon"].get<float>();
        }
        if (paramJson.contains("quantType")) {
            param.normParam.quantType = atb::infer::QuantType(paramJson["quantType"].get<int32_t>());
        }
        if (paramJson.contains("quantInputScale")) {
            param.normParam.quantInputScale = paramJson["quantInputScale"].get<float>();
        }
        if (paramJson.contains("quantInputOffset")) {
            param.normParam.quantInputOffset = paramJson["quantInputOffset"].get<int>();
        }
        if (paramJson.contains("layerNormEps")) {
            param.normParam.layerNormEps = paramJson["layerNormEps"].get<double>();
        }
    } 
    if (param.layerType == atb::infer::RmsNormParam::RMS_NORM_PRENORM) {
        if (paramJson.contains("epsilon")) {
            param.preNormParam.epsilon = paramJson["epsilon"].get<float>();
        }
        if (paramJson.contains("quantType")) {
            param.preNormParam.quantType = atb::infer::QuantType(paramJson["quantType"].get<int32_t>());
        }
        if (paramJson.contains("quantInputScale")) {
            param.preNormParam.quantInputScale = paramJson["quantInputScale"].get<float>();
        }
        if (paramJson.contains("quantInputOffset")) {
            param.preNormParam.quantInputOffset = paramJson["quantInputOffset"].get<int>();
        }
    }  
    return CreateOperation(param, op);
}

static atb::Status StridedBatchMatmulOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::StridedBatchMatmulParam param;
    if (paramJson.contains("transA")) {
        param.transposeA = paramJson["transA"].get<int32_t>();
        ATB_LOG(INFO) << "param.transposeA:" << param.transposeA;
    }
    if (paramJson.contains("transB")) {
        param.transposeB = paramJson["transB"].get<int32_t>();
        ATB_LOG(INFO) << "param.transposeB:" << param.transposeB;
    }
    if (paramJson.contains("batch")) {
        param.batch = paramJson["batch"].get<int32_t>();
        ATB_LOG(INFO) << "param.batch:" << param.batch;
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
        ATB_LOG(INFO) << "param.headNum:" << param.headNum;
    }
    for (auto item : paramJson["m"]) {
        param.m.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["n"]) {
        param.n.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["k"]) {
        param.k.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["lda"]) {
        param.lda.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["ldb"]) {
        param.ldb.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["ldc"]) {
        param.ldc.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["strideA"]) {
        param.strideA.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["strideB"]) {
        param.strideB.push_back(item.get<int32_t>());
    }
    for (auto item : paramJson["strideC"]) {
        param.strideC.push_back(item.get<int32_t>());
    }
    return CreateOperation(param, op);
}

static atb::Status AsStridedOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::AsStridedParam param;
    for (auto item : paramJson["size"]) {
        param.size.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["stride"]) {
        param.stride.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["offset"]) {
        param.offset.push_back(item.get<int64_t>());
    }
    return CreateOperation(param, op);
}

static atb::Status MultinomialOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::MultinomialParam param;
    param.numSamples = paramJson["numSamples"].get<uint32_t>();
    param.randSeed = paramJson["randSeed"].get<uint32_t>();
    return CreateOperation(param, op);
}

static atb::Status ReduceOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::ReduceParam param;
    param.reduceType = paramJson["reduceType"].get<atb::infer::ReduceParam::ReduceType>();
    for (auto item : paramJson["axis"]) {
        param.axis.push_back(item.get<int64_t>());
    }
    return CreateOperation(param, op);
}

static atb::Status WhereOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::WhereParam param;
    ATB_LOG(INFO) << "WhereParam: NULL";
    return CreateOperation(param, op);
}

static atb::Status TransdataOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::infer::TransdataParam param;
    if (paramJson.contains("transdataType")) {
        param.transdataType = atb::infer::TransdataParam::TransdataType(paramJson["transdataType"].get<int>());
    }   
    if (paramJson.contains("outCrops")) {
        param.outCrops.clear();
        for (auto item : paramJson["outCrops"]) {
            param.outCrops.push_back(item.get<int64_t>());
        }
    }
    ATB_LOG(INFO) << "TransdataParam transdataType:" << param.transdataType << ", outCrops:" << param.outCrops;
    return CreateOperation(param, op);
}

static atb::Status FastSoftMaxOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::FastSoftMaxParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    if (paramJson.contains("qSeqLen")) {
        for (auto item : paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    return CreateOperation(param, op);
}

static atb::Status FastSoftMaxGradOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::FastSoftMaxGradParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    if (paramJson.contains("qSeqLen")) {
        for (auto item: paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    return CreateOperation(param, op);
}

static atb::Status FlashAttentionOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::FlashAttentionParam param;
    if (paramJson.contains("scaleValue")) {
        param.scaleValue = paramJson["scaleValue"].get<float>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int64_t>();
    }
    if (paramJson.contains("keepProb")) {
        param.keepProb = paramJson["keepProb"].get<float>();
    }
    if (paramJson.contains("preTokens")) {
        param.preTokens = paramJson["preTokens"].get<int64_t>();
    }
    if (paramJson.contains("nextTokens")) {
        param.nextTokens = paramJson["nextTokens"].get<int64_t>();
    }
    if (paramJson.contains("preciseMode")) {
        param.preciseMode = paramJson["preciseMode"].get<int64_t>();
    }
    if (paramJson.contains("groups")) {
        param.groups = paramJson["groups"].get<int64_t>();
    }
    if (paramJson.contains("ioLayout")) {
        param.ioLayout = paramJson["ioLayout"].get<atb::train::FlashAttentionParam::IoLayout>();
    }
    return CreateOperation(param, op);
}

static atb::Status FlashAttentionBackwardOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::FlashAttentionBackwardParam param;
    if (paramJson.contains("scaleValue")) {
        param.scaleValue = paramJson["scaleValue"].get<float>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int64_t>();
    }
    if (paramJson.contains("keepProb")) {
        param.keepProb = paramJson["keepProb"].get<float>();
    }
    if (paramJson.contains("preTokens")) {
        param.preTokens = paramJson["preTokens"].get<int64_t>();
    }
    if (paramJson.contains("nextTokens")) {
        param.nextTokens = paramJson["nextTokens"].get<int64_t>();
    }
    if (paramJson.contains("preciseMode")) {
        param.preciseMode = paramJson["preciseMode"].get<int64_t>();
    }
    if (paramJson.contains("groups")) {
        param.groups = paramJson["groups"].get<int64_t>();
    }
    if (paramJson.contains("ioLayout")) {
        param.ioLayout = paramJson["ioLayout"].get<atb::train::FlashAttentionBackwardParam::IoLayout>();
    }
    return CreateOperation(param, op);
}

static atb::Status UnpadWithHiddenStateOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::UnpadWithHiddenStateParam param;
    if (paramJson.contains("qSeqLen")) {
        for (auto item : paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("maxSeqLen")) {
        param.maxSeqLen = paramJson["maxSeqLen"].get<int32_t>();
    }
    return CreateOperation(param, op);
}

static atb::Status PadWithHiddenStateOperationCreate(const nlohmann::json &paramJson, atb::Operation **op)
{
    atb::train::PadWithHiddenStateParam param;
    if (paramJson.contains("qSeqLen")) {
        for (auto item : paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("maxSeqLen")) {
        param.maxSeqLen = paramJson["maxSeqLen"].get<int32_t>();
    }
    return CreateOperation(param, op);
}

std::map<std::string, OperationCreateFunc> g_funcMap = {
    {"AllReduceOperation", &AllReduceOperationCreate},
    {"BroadcastOperation", &BroadcastOperationCreate},
    {"AllGatherOperation", &AllGatherOperationCreate},
    {"GatherOperation", &GatherOperationCreate},
    {"ConcatOperation", &ConcatOperationCreate},
    {"SplitOperation", &SplitOperationCreate},
    {"CumsumOperation", &CumsumOperationCreate},
    {"TransposeOperation", &TransposeOperationCreate},
    {"SelfAttentionOperation", &SelfAttentionOperationCreate},
    {"SetValueOperation", &SetValueOperationCreate},
    {"SortOperation", &SortOperationCreate},
    {"ActivationOperation", &ActivationOperationCreate},
    {"KvCacheOperation", &KvCacheOperationCreate},
    {"PagedAttentionOperation", &PagedAttentionOperationCreate},
    {"ReshapeAndCacheOperation", &ReshapeAndCacheOperationCreate},
    {"RopeOperation", &RopeOperationCreate},
    {"FillOperation", &FillOperationCreate},
    {"RepeatOperation", &RepeatOperationCreate},
    {"SliceOperation", &SliceOperationCreate},
    {"SoftmaxOperation", &SoftmaxOperationCreate},
    {"ElewiseOperation", &ElewiseOperationCreate},
    {"TopkToppSamplingOperation", &TopkToppSamplingOperationCreate},
    {"LinearOperation", &LinearOperationCreate},
    {"LinearActivationOperation", &LinearActivationOperationCreate},
    {"LinearActivationQuantOperation", &LinearActivationQuantOperationCreate},
    {"LinearQuantOperation", &LinearQuantOperationCreate},
    {"LinearParallelOperation", &LinearParallelOperationCreate},
    {"LinearSparseOperation", &LinearSparseOperationCreate},
    {"PadOperation", &PadOperationCreate},
    {"UnpadOperation", &UnpadOperationCreate},
    {"MatmulOperation", &MatmulOperationCreate},
    {"GenAttentionMaskOperation", &GenAttentionMaskOperationCreate},
    {"LayerNormOperation", &LayerNormOperationCreate},
    {"RmsNormOperation", &RmsNormOperationCreate},
    {"RmsNormBackwardOperation", &RmsNormBackwardOperationCreate},
    {"RopeGradOperation", &RopeGradOperationCreate},
    {"AsStridedOperation", &AsStridedOperationCreate},
    {"MultinomialOperation", &MultinomialOperationCreate},
    {"ReduceOperation", &ReduceOperationCreate},
    {"WhereOperation", &WhereOperationCreate},
    {"TransdataOperation", &TransdataOperationCreate},
    {"FastSoftMaxOperation", &FastSoftMaxOperationCreate},
    {"FastSoftMaxGradOperation", &FastSoftMaxGradOperationCreate},
    {"StridedBatchMatmulOperation", &StridedBatchMatmulOperationCreate},
    {"FlashAttentionOperation", &FlashAttentionOperationCreate},
    {"FlashAttentionBackwardOperation", &FlashAttentionBackwardOperationCreate},
    {"UnpadWithHiddenStateOperation", &UnpadWithHiddenStateOperationCreate},
    {"PadWithHiddenStateOperation", &PadWithHiddenStateOperationCreate},
};

atb::Status CreateOperation(const std::string &opName, const std::string &param, atb::Operation **operation)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);

    auto it = g_funcMap.find(opName);
    if (it == g_funcMap.end()) {
        ATB_LOG(ERROR) << "not support opName:" << opName;
        return atb::ERROR_INVALID_PARAM;
    }
    try {
        return it->second(paramJson, operation);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << opName << " parse json fail, error:" << e.what();
    }
    return atb::ERROR_INTERNAL_ERROR;
}
