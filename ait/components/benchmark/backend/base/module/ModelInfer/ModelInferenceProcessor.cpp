/*
 * Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
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
 */

#include "Base/ModelInfer/ModelInferenceProcessor.h"
#include "acl/acl.h"
#include "Base/Log/Log.h"

namespace Base {
APP_ERROR ModelInferenceProcessor::GetModelDescInfo()
{
    TensorDesc info;
    int datatype;
    // create in tensos desc info
    size_t numInputs = processModel->GetNumInputs();
    modelDesc_.inTensorsDesc.clear();
    modelDesc_.inTensorsDesc.reserve(numInputs);
    int index = 0;
    for (size_t i = 0; i < numInputs; i++) {
        // dynamicindex not as intensors, it is args info
        if (i == dynamicIndex_) {
            continue;
        }
        // 动态AIPP输入不被录入常规输入
        if (dymAIPPIndexSet_.count(i) != 0) {
            continue;
        }
        CHECK_RET_EQ(
            processModel->GetInTensorDesc(i, info.name, datatype, info.format, info.shape, info.size),
            SUCCESS);
        info.realsize = info.size;
        info.datatype = (TensorDataType)datatype;
        modelDesc_.inTensorsDesc.push_back(info);
        modelDesc_.innames2Index[info.name] = index;
        index++;
    }

    // create out tensos info
    size_t numOutputs = processModel->GetNumOutputs();
    modelDesc_.outTensorsDesc.clear();
    modelDesc_.outTensorsDesc.reserve(numOutputs);
    for (size_t i = 0; i < numOutputs; i++) {
        CHECK_RET_EQ(
            processModel->GetOutTensorDesc(i, info.name, datatype, info.format, info.shape, info.size),
            SUCCESS);
        info.realsize = info.size;
        info.datatype = (TensorDataType)datatype;
        modelDesc_.outTensorsDesc.push_back(info);
        modelDesc_.outnames2Index[info.name] = i;
    }

    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::Init(
    const std::string& modelPath,
    std::shared_ptr<SessionOptions> options,
    const int32_t &deviceId
)
{
    options_ = options;
    deviceId_ = deviceId;

    SETLOGLEVEL(options_->log_level);

    try {
        // make_shared必然会抛出异常
        processModel = std::make_shared<ModelProcess>();
        dyAippCfg = std::make_shared<DynamicAippConfig>();
    } catch (...) {
        return APP_ERR_ACL_BAD_ALLOC;
    }

    // initResource
    CHECK_RET_EQ(processModel->LoadModelFromFile(modelPath), SUCCESS);

    CHECK_RET_EQ(processModel->CreateDesc(), SUCCESS);

    CHECK_RET_EQ(processModel->GetDynamicGearCount(dym_gear_count_), SUCCESS);

    processModel->GetDynamicIndex(dynamicIndex_);

    CHECK_RET_EQ(AllocDymAIPPIndexMem(), APP_ERR_OK);

    CHECK_RET_EQ(GetModelDescInfo(), APP_ERR_OK);

    CHECK_RET_EQ(AllocDyIndexMem(), APP_ERR_OK);

    if (options_->log_level == LOG_DEBUG_LEVEL) {
        processModel->PrintDesc();
    }

    processModel->SetExceptionCallBack();
    return APP_ERR_OK;
}

/*
 * @description Unload Model
 * @return APP_ERROR error code
 */
APP_ERROR ModelInferenceProcessor::DeInit(void)
{
    FreeDyIndexMem();
    FreeDymInfoMem();
    DestroyInferCacheData();
    processModel.reset();
    dyAippCfg.reset();
    FreeDymAIPPMem();
    return APP_ERR_OK;
}

const std::vector<Base::TensorDesc>& ModelInferenceProcessor::GetInputs() const
{
    return modelDesc_.inTensorsDesc;
}

const std::vector<Base::TensorDesc>& ModelInferenceProcessor::GetOutputs() const
{
    return modelDesc_.outTensorsDesc;
}

std::shared_ptr<SessionOptions> ModelInferenceProcessor::GetOptions()
{
    return options_;
}

APP_ERROR ModelInferenceProcessor::DestroyOutMemoryData(std::vector<MemoryData>& outputs)
{
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].ptrData != nullptr) {
            outputs[i].free(outputs[i].ptrData);
        }
    }
    outputs.clear();
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::CreateOutMemoryData(std::vector<MemoryData>& outputs)
{
    size_t size;
    size_t customIndex = 0;
    for (size_t i = 0; i < modelDesc_.outTensorsDesc.size(); ++i) {
        size = modelDesc_.outTensorsDesc[i].size;
        if (customIndex < customOutTensorSize_.size()) {
            size = customOutTensorSize_[customIndex++];
        }
        if (size == 0) {
            ERROR_LOG("out i:%zu size is zero", i);
            return APP_ERR_INFER_OUTPUTSIZE_IS_ZERO;
        }
        DEBUG_LOG("Create OutMemory i:%zu name:%s size:%zu", i, modelDesc_.outTensorsDesc[i].name.c_str(), size);
        Base::MemoryData memorydata(size, MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
        auto ret = MemoryHelper::MxbsMalloc(memorydata);
        if (ret != APP_ERR_OK) {
            ERROR_LOG("MemoryHelper::MxbsMalloc failed.i:%zu name:%s size:%zu ret:%d", \
                      i, modelDesc_.outTensorsDesc[i].name.c_str(), size, ret);
            return ret;
        }
        outputs.push_back(std::move(memorydata));
    }

    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::AddOutTensors(
    std::vector<MemoryData>& outputs,
    std::vector<std::string> outputNames,
    std::vector<TensorBase>& outputTensors
)
{
    bool is_dymshape = (dynamicInfo_.dynamicType == DYNAMIC_SHAPE ? true : false);
    size_t realLen;
    for (const auto& name : outputNames) {
        auto index = modelDesc_.outnames2Index[name];

        std::vector<int64_t> i64shape;
        std::vector<uint32_t> u32shape;
        realLen = processModel->GetOutTensorLen(index, is_dymshape);
        if (processModel->GetCurOutputShape(index, is_dymshape, i64shape) != SUCCESS) {
            // 针对于动态shape场景 如果无法获取真实的输出shape 先填写一个一维的值 以便后续内存可以导出
            auto tmpDataType = static_cast<aclDataType>(modelDesc_.outTensorsDesc[index].datatype);
            i64shape.push_back(realLen / aclDataTypeSize(tmpDataType));
        }
        DEBUG_LOG("AddOutTensors name:%s index:%zu len:%zu outdescsize:%zu shapesize:%zu",
            name.c_str(), index, realLen, modelDesc_.outTensorsDesc[index].size, i64shape.size());
        outputs[index].size = realLen;
        bool isBorrowed = false;
        for (size_t j = 0; j < i64shape.size(); ++j) {
            u32shape.push_back((uint32_t)(i64shape[j]));
        }
        TensorBase outputTensor(outputs[index], isBorrowed, u32shape, modelDesc_.outTensorsDesc[index].datatype);
        outputTensors.push_back(outputTensor);
        // mem control by outTensors so outputs mems nullptr
        outputs[index].ptrData = nullptr;
    }

    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::CheckInVectorAndFillBaseTensor(
    const std::vector<BaseTensor>& feeds,
    std::vector<BaseTensor> &inputs
)
{
    for (size_t i = 0; i < feeds.size(); ++i) {
        BaseTensor baseTensor = {};
        baseTensor.buf = feeds[i].buf;
        baseTensor.size = feeds[i].size;
        if (baseTensor.size != modelDesc_.inTensorsDesc[i].realsize) {
            ERROR_LOG("Check i:%zu name:%s in size:%zu needsize:%zu not match",
                i, modelDesc_.inTensorsDesc[i].name.c_str(), baseTensor.size, modelDesc_.inTensorsDesc[i].realsize);
            return APP_ERR_ACL_FAILURE;
        }
        inputs.push_back(std::move(baseTensor));
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::Inference(
    const std::vector<BaseTensor>& feeds,
    std::vector<std::string> &outputNames,
    std::vector<TensorBase>& outputTensors
)
{
    APP_ERROR ret;
    // create basetensors
    std::vector<BaseTensor> inputs;
    ret = CheckInVectorAndFillBaseTensor(feeds, inputs);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("Check InVector failed ret:%d", ret);
        return ret;
    }
    ret = ModelInference_Inner(inputs, outputNames, outputTensors);
    DestroyInferCacheData();
    return ret;
}

APP_ERROR ModelInferenceProcessor::FirstInference(
    const std::vector<BaseTensor>& feeds,
    std::vector<std::string> &outputNames,
    std::vector<TensorBase>& outputTensors
)
{
    APP_ERROR ret;
    // create basetensors
    std::vector<BaseTensor> inputs;
    ret = CheckInVectorAndFillBaseTensor(feeds, inputs);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("Check InVector failed ret:%d", ret);
        return ret;
    }
    ret = FirstInferenceInner(inputs, outputNames, outputTensors);

    return ret;
}


APP_ERROR ModelInferenceProcessor::CheckInMapAndFillBaseTensor(
    const std::map<std::string,
    TensorBase>& feeds,
    std::vector<BaseTensor> &inputs
)
{
    if (feeds.size() != modelDesc_.inTensorsDesc.size()) {
        ERROR_LOG("intensors size:%zu need size:%zu not match", feeds.size(), modelDesc_.inTensorsDesc.size());
        return APP_ERR_ACL_FAILURE;
    }

    for (size_t i = 0; i < modelDesc_.inTensorsDesc.size(); ++i) {
        auto iter = feeds.find(modelDesc_.inTensorsDesc[i].name);
        if (feeds.end() == iter) {
            ERROR_LOG("intensors i:%zu name:%s not find", i, modelDesc_.inTensorsDesc[i].name.c_str());
            return APP_ERR_ACL_FAILURE;
        }

        BaseTensor baseTensor = {};
        baseTensor.buf = iter->second.GetBuffer();
        baseTensor.size = iter->second.GetByteSize();
        if (baseTensor.size != modelDesc_.inTensorsDesc[i].realsize) {
            ERROR_LOG("Check i:%zu name:%s in size:%zu needsize:%zu not match",
                i, modelDesc_.inTensorsDesc[i].name.c_str(), baseTensor.size, modelDesc_.inTensorsDesc[i].realsize);
            return APP_ERR_ACL_FAILURE;
        }
        inputs.push_back(std::move(baseTensor));
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::Inference(
    const std::map<std::string, TensorBase>& feeds,
    std::vector<std::string> outputNames,
    std::vector<TensorBase>& outputTensors
)
{
    APP_ERROR ret;

    // create basetensors
    std::vector<BaseTensor> inputs;
    ret = CheckInMapAndFillBaseTensor(feeds, inputs);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("Check InVector failed ret:%d", ret);
        return ret;
    }
    ret = ModelInference_Inner(inputs, outputNames, outputTensors);
    return ret;
}

APP_ERROR ModelInferenceProcessor::CheckInVectorAndFillBaseTensor(
    const std::vector<TensorBase>& feeds,
    std::vector<BaseTensor> &inputs
)
{
    for (size_t i = 0; i < feeds.size(); ++i) {
        BaseTensor baseTensor = {};
        baseTensor.buf = feeds[i].GetBuffer();
        baseTensor.size = feeds[i].GetByteSize();
        if (baseTensor.size != modelDesc_.inTensorsDesc[i].realsize) {
            ERROR_LOG("Check i:%zu name:%s in size:%zu needsize:%zu not match",
                i, modelDesc_.inTensorsDesc[i].name.c_str(), baseTensor.size, modelDesc_.inTensorsDesc[i].realsize);
            return APP_ERR_ACL_FAILURE;
        }
        inputs.push_back(std::move(baseTensor));
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::Inference(
    const std::vector<TensorBase>& feeds,
    std::vector<std::string> outputNames,
    std::vector<TensorBase>& outputTensors
)
{
    APP_ERROR ret;
    // create basetensors
    std::vector<BaseTensor> inputs;
    ret = CheckInVectorAndFillBaseTensor(feeds, inputs);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("Check InVector failed ret:%d", ret);
        return ret;
    }
    ret = ModelInference_Inner(inputs, outputNames, outputTensors);
    return ret;
}

APP_ERROR ModelInferenceProcessor::DestroyInferCacheData()
{
    DestroyOutMemoryData(outputsMemDataQue_);
    processModel->DestroyInput(false);
    processModel->DestroyOutput(false);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::UpdateInputsData(const std::vector<int> &inOutRelation, const bool mem_copy)
{
    Result result;
    if (mem_copy) {
        result = processModel->UpdateInputsMemcpy(inOutRelation);
    } else {
        result = processModel->UpdateInputsReuse(inOutRelation);
    }
    if (result != SUCCESS) {
        ERROR_LOG("create inputdataset failed:%d", result);
        return APP_ERR_ACL_FAILURE;
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetInputsData(std::vector<BaseTensor> &inputs)
{
    APP_ERROR ret;

    DestroyInferCacheData();

    if (inputs.size() != modelDesc_.inTensorsDesc.size()) {
        WARN_LOG("intensors in:%zu need:%zu not match", inputs.size(), modelDesc_.inTensorsDesc.size());
        return APP_ERR_ACL_FAILURE;
    }

    if (dynamicInfo_.dynamicType != DYNAMIC_DIMS && dym_gear_count_ > 0) {
        WARN_LOG("check failed dym gearcount:%zu but dymtype:%d not set", dym_gear_count_, dynamicInfo_.dynamicType);
        return APP_ERR_ACL_FAILURE;
    }

    // add dynamic index tensor
    if (dynamicIndex_ != size_t(-1)) {
        Base::BaseTensor dyIndexTensor = {};
        dyIndexTensor.buf = dynamicIndexMemory_.ptrData;
        dyIndexTensor.size = dynamicIndexMemory_.size;
        inputs.insert(inputs.begin() + dynamicIndex_, dyIndexTensor);
    }

    if (dymAIPPIndexSet_.size() != 0) {
        for (const auto& aippSetIt : dymAIPPIndexSet_) {
            Base::BaseTensor dyIndexTensor = {};
            dyIndexTensor.buf = dymAIPPIndexMemory_[aippSetIt.first].ptrData;
            dyIndexTensor.size = dymAIPPIndexMemory_[aippSetIt.first].size;
            inputs.insert(inputs.begin() + aippSetIt.first, dyIndexTensor);
        }
    }

    // create output memdata
    ret = CreateOutMemoryData(outputsMemDataQue_);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("create outmemory data failed:%d", ret);
        return ret;
    }

    // add data to input dataset
    for (const auto& tensor : inputs) {
        auto result = processModel->CreateInput(tensor.buf, tensor.size);
        if (result != SUCCESS) {
            ERROR_LOG("create inputdataset failed:%d", result);
            return APP_ERR_ACL_FAILURE;
        }
    }

    // add data to output dataset
    for (const auto& tensor : outputsMemDataQue_) {
        auto result = processModel->CreateOutput(tensor.ptrData, tensor.size);
        if (result != SUCCESS) {
            ERROR_LOG("create outputdataset failed:%d", result);
            return APP_ERR_ACL_FAILURE;
        }
    }

    ret = SetDynamicInfo();
    if (ret != APP_ERR_OK) {
        ERROR_LOG("set dynamic info failed:%d", ret);
        return ret;
    }

    DEBUG_LOG("SetInputData successfully");
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetAippConfigData()
{
    if (dyAippCfg->IsActivated() && dyAippCfg->ModelIsLegal()) {
        // 读取合法的config文件，且模型有一个动态aipp输入才进行aipp参数的具体设置
        DEBUG_LOG("SetInputAIPP start");
        for (auto& aippSetIt : dymAIPPIndexSet_) {
            Result result = processModel->SetInputAIPP(aippSetIt.first, aippSetIt.second);
            if (result != SUCCESS) {
                ERROR_LOG("ModelProcess::SetInputAIPP failed. index:%d result:%d ", int(aippSetIt.first), result);
                return APP_ERR_ACL_FAILURE;
            }
        }
        DEBUG_LOG("SetInputAIPP successfully");
    } else if ((!dyAippCfg->IsActivated()) && dyAippCfg->ModelIsLegal()) {
        // 模型有一个动态aipp输入，但是没有读取到合法的配置文件
        ERROR_LOG("model with dynamic aipp input can't find config file.");
        return APP_ERR_ACL_FAILURE;
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::GetOutputs(
    std::vector<std::string> outputNames,
    std::vector<TensorBase> &outputTensors
)
{
    for (const auto& name : outputNames) {
        if (modelDesc_.outnames2Index.find(name) == modelDesc_.outnames2Index.end()) {
            ERROR_LOG("outnames %s not valid", name.c_str());
            return APP_ERR_ACL_FAILURE;
        }
    }

    APP_ERROR ret = AddOutTensors(outputsMemDataQue_, outputNames, outputTensors);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("create outTensor failed ret:%d", ret);
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::RepeatInference(
    const std::vector<int>& inOutRelation,
    std::vector<std::string> &outputNames,
    std::vector<TensorBase>& outputTensors,
    const bool get_outputs, const bool mem_copy
)
{
    APP_ERROR ret = UpdateInputsData(inOutRelation, mem_copy);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("UpdateInputsData failed ret:%d", ret);
        return ret;
    }
    int loopTimes;
    if (mem_copy) {
        loopTimes = options_->loop;
    } else {
        loopTimes = 1;
    }
    for (int i = 0; i < loopTimes; i++) {
        ret = Execute();
        if (ret != APP_ERR_OK) {
            ERROR_LOG("Execute Infer failed ret:%d", ret);
            return ret;
        }
        if (loopTimes > 1) {
            printf("\rloop inference exec: (%d/%d)", i + 1, loopTimes);
            fflush(stdout);
        }
    }
    if (loopTimes > 1) {
        printf("\n");
    }
    if (get_outputs) {
        ret = GetOutputs(outputNames, outputTensors);
        if (ret != APP_ERR_OK) {
            ERROR_LOG("Get OutTensors failed ret:%d", ret);
            return ret;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::FirstInferenceInner(
    std::vector<BaseTensor> &inputs,
    std::vector<std::string> outputNames,
    std::vector<TensorBase>& outputTensors
)
{
    APP_ERROR ret = SetInputsData(inputs);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("Set InputsData failed ret:%d", ret);
        return ret;
    }
    if (dyAippCfg->ModelIsLegal()) {
        ret = SetAippConfigData();
        if (ret != APP_ERR_OK) {
            ERROR_LOG("Set AippConfigData failed ret:%d", ret);
            return ret;
        }
        if (options_->loop > 1) {
            printf("\n");
        }
    }
    processModel->InitReuseOutput();
    for (int i = 0; i < options_->loop; i++) {
        ret = Execute();
        if (ret != APP_ERR_OK) {
            ERROR_LOG("Execute Infer failed ret:%d", ret);
            return ret;
        }
        if (options_->loop > 1) {
            printf("\rloop inference exec: (%d/%d)", i + 1, options_->loop);
            fflush(stdout);
        }
    }
    if (options_->loop > 1) {
        printf("\n");
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::ModelInference_Inner(
    std::vector<BaseTensor> &inputs,
    std::vector<std::string> outputNames,
    std::vector<TensorBase>& outputTensors
)
{
    APP_ERROR ret = SetInputsData(inputs);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("Set InputsData failed ret:%d", ret);
        return ret;
    }
    if (dyAippCfg->ModelIsLegal()) {
        ret = SetAippConfigData();
        if (ret != APP_ERR_OK) {
            ERROR_LOG("Set AippConfigData failed ret:%d", ret);
            return ret;
        }
        if (options_->loop > 1) {
            printf("\n");
        }
    }
    for (int i = 0; i < options_->loop; i++) {
        ret = Execute();
        if (ret != APP_ERR_OK) {
            ERROR_LOG("Execute Infer failed ret:%d", ret);
            return ret;
        }
        if (options_->loop > 1) {
            printf("\rloop inference exec: (%d/%d)", i + 1, options_->loop);
            fflush(stdout);
        }
    }
    if (options_->loop > 1) {
        printf("\n");
    }
    ret = GetOutputs(outputNames, outputTensors);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("Get OutTensors failed ret:%d", ret);
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::Execute()
{
    struct timeval start = { 0 };
    struct timeval end = { 0 };
    gettimeofday(&start, nullptr);

    Result result = processModel->Execute();
    if (result != SUCCESS) {
        ERROR_LOG("acl execute failed:%d", result);
        return APP_ERR_ACL_FAILURE;
    }

    gettimeofday(&end, nullptr);
    float time_cost = 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000.000;
    DEBUG_LOG("model aclExec cost : %f", time_cost);
    sumaryInfo_.execTimeList.push_back(time_cost);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::ResetSumaryInfo()
{
    sumaryInfo_.execTimeList.clear();
    return APP_ERR_OK;
}

const InferSumaryInfo& ModelInferenceProcessor::GetSumaryInfo()
{
    return sumaryInfo_;
}

APP_ERROR ModelInferenceProcessor::AllocDymAIPPIndexMem()
{
    std::vector<size_t> dymAIPPIndexList_ = {};
    processModel->GetAIPPIndexList(dymAIPPIndexList_);
    if (dymAIPPIndexList_.size() == 0) {
        return APP_ERR_OK;
    }
    if (dymAIPPIndexList_.size() != 0 && dymAIPPIndexMemory_.size() != 0) {
        return APP_ERR_OK;
    }

    for (const auto& index : dymAIPPIndexList_) {
        TensorDesc info;
        int datatype;
        CHECK_RET_EQ(
            processModel->GetInTensorDesc(index, info.name, datatype, info.format, info.shape, info.size),
            SUCCESS);
        MemoryData memdata;
        memdata.size = info.size;
        memdata.type = MemoryData::MemoryType::MEMORY_DEVICE;
        memdata.deviceId = deviceId_;
        DEBUG_LOG("lcm debug aipp config index:%d allow size:%d\n", int(index), int(info.size));
        auto ret = MemoryHelper::MxbsMalloc(memdata);
        if (ret != APP_ERR_OK) {
            ERROR_LOG("MemoryHelper::MxbsMalloc failed. ret=%d", ret);
            return ret;
        }
        dymAIPPIndexMemory_[index] = memdata;
        dymAIPPIndexSet_[index] = nullptr;
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::FreeDymAIPPMem()
{
    for (auto& aippSetIt : dymAIPPIndexSet_) {
        if (aippSetIt.second != nullptr) {
            processModel->FreeAIPP(aippSetIt.second);
            aippSetIt.second = nullptr;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::AllocDyIndexMem()
{
    if (dynamicIndex_ == (size_t)-1 || dynamicIndexMemory_.ptrData != nullptr) {
        return APP_ERR_OK;
    }

    TensorDesc info;
    int datatype;
    CHECK_RET_EQ(
        processModel->GetInTensorDesc(dynamicIndex_, info.name, datatype, info.format, info.shape, info.size),
        SUCCESS);

    dynamicIndexMemory_.size = info.size;
    dynamicIndexMemory_.type = MemoryData::MemoryType::MEMORY_DEVICE;
    dynamicIndexMemory_.deviceId = deviceId_;
    auto ret = MemoryHelper::MxbsMalloc(dynamicIndexMemory_);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("MemoryHelper::MxbsMalloc failed. ret=%d", ret);
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::FreeDyIndexMem()
{
    if (dynamicIndexMemory_.ptrData != nullptr) {
        dynamicIndexMemory_.free(dynamicIndexMemory_.ptrData);
        dynamicIndexMemory_.ptrData = nullptr;
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::FreeDymInfoMem()
{
    switch (dynamicInfo_.dynamicType) {
        case DYNAMIC_DIMS:
            if (dynamicInfo_.dyDims.pDims != nullptr) {
                free(dynamicInfo_.dyDims.pDims);
                dynamicInfo_.dyDims.pDims = nullptr;
            }
            break;
        case DYNAMIC_SHAPE:
            if (dynamicInfo_.dyShape.pShapes != nullptr) {
                free(dynamicInfo_.dyShape.pShapes);
                dynamicInfo_.dyShape.pShapes = nullptr;
            }
            break;
        default:
            break;
    }

    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetStaticBatch()
{
    FreeDymInfoMem();
    dynamicInfo_.dynamicType = STATIC_BATCH;
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetDynamicBatchsize(int batchsize)
{
    bool is_dymbatch = false;

    FreeDymInfoMem();

    CHECK_RET_EQ(processModel->CheckDynamicBatchSize(batchsize, is_dymbatch), SUCCESS);
    CHECK_RET_EQ(processModel->GetMaxBatchSize(dynamicInfo_.dyBatch.maxbatchsize), SUCCESS);

    for (size_t i = 0; i < modelDesc_.inTensorsDesc.size(); ++i) {
        auto tensorBegin = modelDesc_.inTensorsDesc[i].shape.begin();
        auto tensorEnd = modelDesc_.inTensorsDesc[i].shape.end();
        if (find(tensorBegin, tensorEnd, -1) != tensorEnd) {
            modelDesc_.inTensorsDesc[i].realsize = modelDesc_.inTensorsDesc[i].size *
                batchsize / dynamicInfo_.dyBatch.maxbatchsize;
        }
    }

    dynamicInfo_.dyBatch.batchSize = batchsize;
    dynamicInfo_.dynamicType = DYNAMIC_BATCH;
    return APP_ERR_OK;
}

uint64_t ModelInferenceProcessor::GetMaxDymBatchsize()
{
    uint64_t maxBatchSize = 0;
    CHECK_RET_EQ(processModel->GetMaxBatchSize(maxBatchSize), SUCCESS);
    return maxBatchSize;
}

int ModelInferenceProcessor::GetDymAIPPInputExist()
{
    return processModel->CheckDymAIPPInputExist();
}

APP_ERROR ModelInferenceProcessor::CheckDymAIPPInputExist()
{
    int ret = processModel->CheckDymAIPPInputExist();
    if (ret != 1) {
        return APP_ERR_ACL_FAILURE;
    }
    dyAippCfg->ActivateModel();
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::AippSetMaxBatchSize(uint64_t batchSize)
{
    CHECK_RET_EQ(dyAippCfg->SetMaxBatchSize(batchSize), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetInputFormat(std::string iptFmt)
{
    CHECK_RET_EQ(dyAippCfg->SetInputFormat(iptFmt), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetSrcImageSize(std::vector<int> srcImageSize)
{
    CHECK_RET_EQ(dyAippCfg->SetSrcImageSize(srcImageSize), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetRbuvSwapSwitch(int rsSwitch)
{
    CHECK_RET_EQ(dyAippCfg->SetRbuvSwapSwitch(rsSwitch), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetAxSwapSwitch(int asSwitch)
{
    CHECK_RET_EQ(dyAippCfg->SetAxSwapSwitch(asSwitch), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetCscParams(std::vector<int> cscParams)
{
    CHECK_RET_EQ(dyAippCfg->SetCscParams(cscParams), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetCropParams(std::vector<int> cropParams)
{
    CHECK_RET_EQ(dyAippCfg->SetCropParams(cropParams), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetPaddingParams(std::vector<int> padParams)
{
    CHECK_RET_EQ(dyAippCfg->SetPaddingParams(padParams), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetDtcPixelMean(std::vector<int> meanParams)
{
    CHECK_RET_EQ(dyAippCfg->SetDtcPixelMean(meanParams), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetDtcPixelMin(std::vector<float> minParams)
{
    CHECK_RET_EQ(dyAippCfg->SetDtcPixelMin(minParams), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetPixelVarReci(std::vector<float> reciParams)
{
    CHECK_RET_EQ(dyAippCfg->SetPixelVarReci(reciParams), APP_ERR_OK);
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetDynamicHW(int width, int height)
{
    bool is_dymHW;
    pair<uint64_t, uint64_t> dynamicHW = {width, height};

    FreeDymInfoMem();

    CHECK_RET_EQ(processModel->CheckDynamicHWSize(dynamicHW, is_dymHW), SUCCESS);
    CHECK_RET_EQ(processModel->GetMaxDynamicHWSize(dynamicInfo_.dyHW.maxHWSize), SUCCESS);

    for (size_t i = 0; i < modelDesc_.inTensorsDesc.size(); ++i) {
        auto tensorBegin = modelDesc_.inTensorsDesc[i].shape.begin();
        auto tensorEnd = modelDesc_.inTensorsDesc[i].shape.end();
        if (find(tensorBegin, tensorEnd, -1) != tensorEnd) {
            modelDesc_.inTensorsDesc[i].realsize = modelDesc_.inTensorsDesc[i].size *
                width * height / dynamicInfo_.dyHW.maxHWSize;
        }
    }

    dynamicInfo_.dyHW.imageSize.width = width;
    dynamicInfo_.dyHW.imageSize.height = height;
    dynamicInfo_.dynamicType = DYNAMIC_HW;
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetDynamicDims(std::string dymdimsStr)
{
    // 获取动态维度数量
    CHECK_RET_EQ(processModel->GetDynamicGearCount(dym_gear_count_), SUCCESS);

    FreeDymInfoMem();
    if (dynamicInfo_.dyDims.pDims == nullptr) {
        dynamicInfo_.dyDims.pDims = (DyDimsInfo *)calloc(1, sizeof(DyDimsInfo));
    }

    // 如何释放数组 动态
    aclmdlIODims *dims = new aclmdlIODims[dym_gear_count_];
    Utils::SplitStringSimple(dymdimsStr, dynamicInfo_.dyDims.pDims->dym_dims, ';', ':', ',');

    if (dym_gear_count_ <= 0) {
        printf("the dynamic_dims parameter is not specified for model conversion");
        delete [] dims;
        free(dynamicInfo_.dyDims.pDims);
        return APP_ERR_ACL_FAILURE;
    }

    Result ret =  processModel->CheckDynamicDims(dynamicInfo_.dyDims.pDims->dym_dims, dym_gear_count_, dims);
    if (ret != SUCCESS) {
        ERROR_LOG("check dynamic dims failed, please set correct dymDims paramenter");
        delete [] dims;
        free(dynamicInfo_.dyDims.pDims);
        return APP_ERR_ACL_FAILURE;
    }

    DEBUG_LOG("prepare dynamic dims successful");

    // update realsize according real shapes
    vector<string> dymdims_tmp;
    Utils::SplitStringWithPunctuation(dymdimsStr, dymdims_tmp, ';');

    std::map<string, int64_t> namedimsmap;
    ret = Utils::SplitStingGetNameDimsMulMap(dymdims_tmp, namedimsmap);
    if (ret != SUCCESS) {
        ERROR_LOG("split dims str failed");
        delete [] dims;
        free(dynamicInfo_.dyDims.pDims);
        return APP_ERR_ACL_FAILURE;
    }
    for (auto map : namedimsmap) {
        if (modelDesc_.innames2Index.find(map.first) == modelDesc_.innames2Index.end()) {
            WARN_LOG("find no in names:%s", map.first.c_str());
            continue;
        }
        size_t inindex = modelDesc_.innames2Index[map.first];   // get intensors index by name
        auto tmpDataType = static_cast<aclDataType>(modelDesc_.inTensorsDesc[inindex].datatype);
        modelDesc_.inTensorsDesc[inindex].realsize = map.second * aclDataTypeSize(tmpDataType);
    }

    delete [] dims;

    dynamicInfo_.dynamicType = DYNAMIC_DIMS;
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetDynamicShape(std::string dymshapeStr)
{
    vector<string> dym_shape_tmp;
    Utils::SplitStringWithPunctuation(dymshapeStr, dym_shape_tmp, ';');

    FreeDymInfoMem();
    if (dynamicInfo_.dyShape.pShapes == nullptr) {
        dynamicInfo_.dyShape.pShapes = (DyShapeInfo *)calloc(1, sizeof(DyShapeInfo));
    }

    std::map<string, std::vector<int64_t>> name2shapesmap;
    Result ret = processModel->CheckDynamicShape(
        dym_shape_tmp, name2shapesmap,
        dynamicInfo_.dyShape.pShapes->dims_num);
    if (ret != SUCCESS) {
        ERROR_LOG("check dynamic shape failed");
        free(dynamicInfo_.dyShape.pShapes);
        return APP_ERR_ACL_FAILURE;
    }

    dynamicInfo_.dyShape.pShapes->dym_shape_map = name2shapesmap;

    // update realsize according real shapes
    std::map<string, int64_t> namedimsmap;
    ret = Utils::SplitStingGetNameDimsMulMap(dym_shape_tmp, namedimsmap);
    if (ret != SUCCESS) {
        ERROR_LOG("split dims str failed");
        free(dynamicInfo_.dyShape.pShapes);
        return APP_ERR_ACL_FAILURE;
    }
    for (auto map : namedimsmap) {
        if (modelDesc_.innames2Index.find(map.first) == modelDesc_.innames2Index.end()) {
            WARN_LOG("find no in names:%s", map.first.c_str());
            continue;
        }
        size_t inindex = modelDesc_.innames2Index[map.first];   // get intensors index by name
        auto tmpDataType = static_cast<aclDataType>(modelDesc_.inTensorsDesc[inindex].datatype);
        modelDesc_.inTensorsDesc[inindex].realsize = map.second * aclDataTypeSize(tmpDataType);
    }

    dynamicInfo_.dynamicType = DYNAMIC_SHAPE;
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetCustomOutTensorsSize(std::vector<size_t> customOutSize)
{
    customOutTensorSize_ = customOutSize;
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetDymAIPPInfoSet()
{
    dyAippCfg->ActivateConfig(); // config文件确定合法
    uint64_t MaxBS = dyAippCfg->GetMaxBatchSize();
    DEBUG_LOG("debug now set aipp index list size:%d\n", int(dymAIPPIndexSet_.size()));
    for (auto& aippSetIt : dymAIPPIndexSet_) {
        Result ret = processModel->GetDymAIPPConfigSet(dyAippCfg, aippSetIt.second, MaxBS);
        DEBUG_LOG("debug get aipp config set index:%d\n", int(aippSetIt.first));
        if (ret != SUCCESS) {
            ERROR_LOG("ModelProcess::SetDynamicAippConfig failed.index: %d ret %d", int(aippSetIt.first), ret);
            return APP_ERR_FAILURE;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR ModelInferenceProcessor::SetDynamicInfo()
{
    pair<uint64_t, uint64_t> dynamicHW;
    switch (dynamicInfo_.dynamicType) {
        case DYNAMIC_BATCH:
            CHECK_RET_EQ(processModel->SetDynamicBatchSize(dynamicInfo_.dyBatch.batchSize), SUCCESS);
            break;
        case DYNAMIC_HW:
            dynamicHW = {dynamicInfo_.dyHW.imageSize.width, dynamicInfo_.dyHW.imageSize.height};
            CHECK_RET_EQ(processModel->SetDynamicHW(dynamicHW), SUCCESS);
            break;
        case DYNAMIC_DIMS:
            if (dynamicInfo_.dyDims.pDims == nullptr) {
                WARN_LOG("error dynamic dims type but pdims is null");
            } else {
                CHECK_RET_EQ(processModel->SetDynamicDims(dynamicInfo_.dyDims.pDims->dym_dims), SUCCESS);
            }
            break;
        case DYNAMIC_SHAPE:
            if (dynamicInfo_.dyShape.pShapes == nullptr) {
                WARN_LOG("error dynamic shapes type but pshapes is null");
            } else {
                CHECK_RET_EQ(processModel->SetDynamicShape(
                    dynamicInfo_.dyShape.pShapes->dym_shape_map, dynamicInfo_.dyShape.pShapes->dims_num), SUCCESS);
            }
            break;
        default:
            break;
    }
    return APP_ERR_OK;
}
}  // namespace Base
