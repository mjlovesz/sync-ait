/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendTransformerBoost is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <iostream>
#include <string>
#include <functional>
#include <nlohmann/json.hpp>

#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#include "atb/operation.h"
#include "operation_factory.h"

using CreateOperationFuncPtr = std::function<atb::Operation *(const nlohmann::json &)>;

static atb::Operation *ActivationOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ActivationParam param;
    if (paramJson.contains("activationType")) {
        param.activationType = atb::infer::ActivationType(paramJson["activationType"].get<int32_t>());
    }
    if (paramJson.contains("scale")) {
        param.scale = paramJson["scale"].get<float>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *AllGatherOperationCreate(const nlohmann::json &paramJson)
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
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *AllReduceOperationCreate(const nlohmann::json &paramJson)
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
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *AsStridedOperationCreate(const nlohmann::json &paramJson)
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *BroadcastOperationCreate(const nlohmann::json &paramJson)
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
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *ConcatOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ConcatParam param;
    if (paramJson.contains("concatDim")) {
        param.concatDim = paramJson["concatDim"].get<int>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *CumsumOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::CumsumParam param;
    for (auto item : paramJson["axes"]) {
        param.axes.push_back(item.get<int64_t>());
    }
    if (paramJson.contains("exclusive")) {
        param.exclusive = paramJson["exclusive"].get<bool>();
    }
    if (paramJson.contains("reverse")) {
        param.reverse = paramJson["reverse"].get<bool>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *ElewiseOperationCreate(const nlohmann::json &paramJson)
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *FastSoftMaxGradOperationCreate(const nlohmann::json &paramJson)
{
    atb::train::FastSoftMaxGradParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    if (paramJson.contains("qSeqLen")) {
        for (auto item : paramJson["qSeqLen"]) {
            param.qSeqLen.push_back(item.get<int32_t>());
        }
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *FastSoftMaxOperationCreate(const nlohmann::json &paramJson)
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *FillOperationCreate(const nlohmann::json &paramJson)
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *GatherOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::GatherParam param;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int64_t>();
    }
    if (paramJson.contains("batchDims")) {
        param.batchDims = paramJson["batchDims"].get<int64_t>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *GenAttentionMaskOperationCreate(const nlohmann::json &paramJson)
{
    atb::train::GenAttentionMaskParam param;
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
    }
    for (auto item : paramJson["seqLen"]) {
        param.seqLen.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *IndexAddOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::IndexAddParam param;
    param.indexType = paramJson["indexType"].get<atb::infer::IndexAddParam::IndexType>();
    param.axis = paramJson["axis"].get<int32_t>();
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *KvCacheOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::KvCacheParam param;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *LayerNormOperationCreate(const nlohmann::json &paramJson)
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
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *LinearOperationCreate(const nlohmann::json &paramJson)
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *LinearParallelOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::LinearParallelParam param;
    if (paramJson.find("transWeight") != paramJson.end()) {
        param.transWeight = paramJson["transWeight"].get<bool>();
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
    if (paramJson.find("rankTableFile") != paramJson.end()) {
        param.rankTableFile = paramJson["rankTableFile"].get<std::string>();
    }
    if (paramJson.contains("type")) {
        param.type = atb::infer::LinearParallelParam::ParallelType(paramJson["type"].get<int>());
    }
    if (paramJson.contains("keepIntermediate")) {
        param.keepIntermediate = paramJson["keepIntermediate"].get<bool>();
    }
    param.rank = paramJson["rank"].get<int>();
    param.rankSize = paramJson["rankSize"].get<int>();
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *LinearSparseOperationCreate(const nlohmann::json &paramJson)
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

    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *MultinomialOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::MultinomialParam param;
    param.numSamples = paramJson["numSamples"].get<uint32_t>();
    param.randSeed = paramJson["randSeed"].get<uint32_t>();
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *NonzeroOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::NonzeroParam param;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *OnehotOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::OnehotParam param;
    if (paramJson.contains("axis")) {
        param.axis = paramJson["axis"].get<int64_t>();
    }
    if (paramJson.contains("depth")) {
        param.depth = paramJson["depth"].get<int64_t>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *PadOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::PadParam param;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *PadWithHiddenStateOperationCreate(const nlohmann::json &paramJson)
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *ReduceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ReduceParam param;
    param.reduceType = paramJson["reduceType"].get<atb::infer::ReduceParam::ReduceType>();
    for (auto item : paramJson["axis"]) {
        param.axis.push_back(item.get<int64_t>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *PagedAttentionOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::PagedAttentionParam param;
    param.headNum = paramJson["headNum"].get<int>();
    param.qkScale = paramJson["qkScale"].get<float>();
    param.kvHeadNum = paramJson["kvHeadNum"].get<int>();

    if (paramJson.contains("maskType")) {
        param.maskType = atb::infer::PagedAttentionParam::MaskType(paramJson["maskType"].get<int32_t>());
    }

    if (paramJson.contains("quantType")) {
        param.quantType = atb::infer::PagedAttentionParam::QuantType(paramJson["quantType"].get<int32_t>());
    }

    if (paramJson.contains("hasQuantOffset")) {
        param.hasQuantOffset = paramJson["hasQuantOffset"].get<bool>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *RepeatOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RepeatParam param;
    for (auto item : paramJson["multiples"]) {
        param.multiples.push_back(item.get<int64_t>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *ReshapeAndCacheOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::ReshapeAndCacheParam param;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *RmsNormBackwardOperationCreate(const nlohmann::json &paramJson)
{
    atb::train::RmsNormBackwardParam param;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *RmsNormOperationCreate(const nlohmann::json &paramJson)
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
        if (paramJson.contains("hasBias")) {
            param.preNormParam.hasBias = paramJson["hasBias"].get<bool>();
        }
    }
    if (param.layerType == atb::infer::RmsNormParam::RMS_NORM_POSTNORM) {
        if (paramJson.contains("hasBias")) {
            param.postNormParam.hasBias = paramJson["hasBias"].get<bool>();
        }
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *RopeGradOperationCreate(const nlohmann::json &paramJson)
{
    atb::train::RopeGradParam param;
    for (auto item : paramJson["qSeqLen"]) {
        param.qSeqLen.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *RopeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::RopeParam param;
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
    if (paramJson.contains("cosFormat")) {
        param.cosFormat = paramJson["cosFormat"].get<int>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *SelfAttentionOperationCreate(const nlohmann::json &paramJson)
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
    if (paramJson.contains("kvHeadNum")) {
        param.kvHeadNum = paramJson["kvHeadNum"].get<int>();
    }
    if (paramJson.contains("maskType")) {
        param.maskType = atb::infer::SelfAttentionParam::MaskType(paramJson["maskType"].get<int32_t>());
    }
    if (paramJson.contains("calcType")) {
        param.calcType = atb::infer::SelfAttentionParam::CalcType(paramJson["calcType"].get<int32_t>());
    }
    if (paramJson.contains("clampType")) {
        param.clampType = atb::infer::SelfAttentionParam::ClampType(paramJson["clampType"].get<int32_t>());
    }
    if (paramJson.contains("clampMin")) {
        param.clampMin = paramJson["clampMin"].get<float>();
    }
    if (paramJson.contains("clampMax")) {
        param.clampMax = paramJson["clampMax"].get<float>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *SetValueOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SetValueParam param;
    for (auto item : paramJson["starts"]) {
        param.starts.push_back(item.get<int>());
    }
    for (auto item : paramJson["ends"]) {
        param.ends.push_back(item.get<int>());
    }
    for (auto item : paramJson["strides"]) {
        param.strides.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *SliceOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SliceParam param;
    for (auto item : paramJson["offsets"]) {
        param.offsets.push_back(item.get<int64_t>());
    }
    for (auto item : paramJson["size"]) {
        param.size.push_back(item.get<int64_t>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *SoftmaxOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SoftmaxParam param;
    for (auto item : paramJson["axes"]) {
        param.axes.push_back(item.get<int64_t>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *SortOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SortParam param;
    for (auto item : paramJson["num"]) {
        param.num.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *SplitOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::SplitParam param;
    if (paramJson.contains("splitDim")) {
        param.splitDim = paramJson["splitDim"].get<int>();
    }
    if (paramJson.contains("splitNum")) {
        param.splitNum = paramJson["splitNum"].get<int>();
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *StridedBatchMatmulOperationCreate(const nlohmann::json &paramJson)
{
    atb::train::StridedBatchMatmulParam param;
    if (paramJson.contains("transA")) {
        param.transposeA = paramJson["transA"].get<int32_t>();
    }
    if (paramJson.contains("transB")) {
        param.transposeB = paramJson["transB"].get<int32_t>();
    }
    if (paramJson.contains("batch")) {
        param.batch = paramJson["batch"].get<int32_t>();
    }
    if (paramJson.contains("headNum")) {
        param.headNum = paramJson["headNum"].get<int32_t>();
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *TopkToppSamplingOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TopkToppSamplingParam param;
    param.randSeed = paramJson["randSeed"].get<uint32_t>();
    param.topk = paramJson["topk"].get<uint32_t>();
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *TransdataOperationCreate(const nlohmann::json &paramJson)
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *TransposeOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::TransposeParam param;
    for (auto item : paramJson["perm"]) {
        param.perm.push_back(item.get<int>());
    }
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *UnpadOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::UnpadParam param;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *UnpadWithHiddenStateOperationCreate(const nlohmann::json &paramJson)
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
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

static atb::Operation *WhereOperationCreate(const nlohmann::json &paramJson)
{
    atb::infer::WhereParam param;
    atb::Operation *op;
    CreateOperation(param, &op);
    return op;
}

std::map<std::string, CreateOperationFuncPtr> g_funcMap = {
    { "ActivationOperation", &ActivationOperationCreate },
    { "AllGatherOperation", &AllGatherOperationCreate },
    { "AllReduceOperation", &AllReduceOperationCreate },
    { "AsStridedOperation", &AsStridedOperationCreate },
    { "BroadcastOperation", &BroadcastOperationCreate },
    { "ConcatOperation", &ConcatOperationCreate },
    { "CumsumOperation", &CumsumOperationCreate },
    { "ElewiseOperation", &ElewiseOperationCreate },
    { "FastSoftMaxGradOperation", &FastSoftMaxGradOperationCreate },
    { "FastSoftMaxOperation", &FastSoftMaxOperationCreate },
    { "FillOperation", &FillOperationCreate },
    { "GatherOperation", &GatherOperationCreate },
    { "GenAttentionMaskOperation", &GenAttentionMaskOperationCreate },
    { "IndexAddOperation", &IndexAddOperationCreate },
    { "KvCacheOperation", &KvCacheOperationCreate },
    { "LayerNormOperation", &LayerNormOperationCreate },
    { "LinearOperation", &LinearOperationCreate },
    { "LinearParallelOperation", &LinearParallelOperationCreate },
    { "LinearSparseOperation", &LinearSparseOperationCreate },
    { "MultinomialOperation", &MultinomialOperationCreate },
    { "NonzeroOperation", &NonzeroOperationCreate },
    { "OnehotOperation", &OnehotOperationCreate },
    { "PadOperation", &PadOperationCreate },
    { "PadWithHiddenStateOperation", &PadWithHiddenStateOperationCreate },
    { "ReduceOperation", &ReduceOperationCreate },
    { "PagedAttentionOperation", &PagedAttentionOperationCreate },
    { "RepeatOperation", &RepeatOperationCreate },
    { "ReshapeAndCacheOperation", &ReshapeAndCacheOperationCreate },
    { "RmsNormBackwardOperation", &RmsNormBackwardOperationCreate },
    { "RmsNormOperation", &RmsNormOperationCreate },
    { "RopeGradOperation", &RopeGradOperationCreate },
    { "RopeOperation", &RopeOperationCreate },
    { "SelfAttentionOperation", &SelfAttentionOperationCreate },
    { "SetValueOperation", &SetValueOperationCreate },
    { "SliceOperation", &SliceOperationCreate },
    { "SoftmaxOperation", &SoftmaxOperationCreate },
    { "SortOperation", &SortOperationCreate },
    { "SplitOperation", &SplitOperationCreate },
    { "StridedBatchMatmulOperation", &StridedBatchMatmulOperationCreate },
    { "TopkToppSamplingOperation", &TopkToppSamplingOperationCreate },
    { "TransdataOperation", &TransdataOperationCreate },
    { "TransposeOperation", &TransposeOperationCreate },
    { "UnpadOperation", &UnpadOperationCreate },
    { "UnpadWithHiddenStateOperation", &UnpadWithHiddenStateOperationCreate },
    { "WhereOperation", &WhereOperationCreate },
};

int registerAll() {
    int retVal = 0;
    for (auto & item : g_funcMap) {
        auto ret = atb_speed::OperationFactory::Register(item.first, item.second);  // ret == True for successful
        if (! ret) {
            retVal += 1;
        }
    }
    return retVal;
}

void printAll() {
  auto foo = atb_speed::OperationFactory::GetRegistryMap();
  for (auto & item : foo) {
      std::cout << "Op: " << item.first << std::endl;
  }
}

extern "C" {
    int RegisterAll() { return registerAll(); }
    void PrintAll() { printAll(); }
}
