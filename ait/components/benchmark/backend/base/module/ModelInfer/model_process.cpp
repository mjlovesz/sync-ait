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

#include "model_process.h"
#include <cstddef>
#include "utils.h"
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
bool g_isDevice = true;
bool g_isTxt = false;
vector<int> g_output_size;

int GetDynamicAippParaByBatch(
    size_t batchIndex,
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    std::string cfgItem
)
{
    if (cfgItem.compare("dtcPixelMean") == 0) {
        if (dyAippCfg->GetDtcPixelMean().count(batchIndex) == 1) {
            return batchIndex;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("crop") == 0) {
        if (dyAippCfg->GetCropParams().count(batchIndex) == 1) {
            return batchIndex;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("pad") == 0) {
        if (dyAippCfg->GetPaddingParams().count(batchIndex) == 1) {
            return batchIndex;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("dtcPixelMin") == 0) {
        if (dyAippCfg->GetDtcPixelMin().count(batchIndex) == 1) {
            return batchIndex;
        } else {
            return -1;
        }
    } else if (cfgItem.compare("pixelVarReci") == 0) {
        if (dyAippCfg->GetPixelVarReci().count(batchIndex) == 1) {
            return batchIndex;
        } else {
            return -1;
        }
    }

    return -1;
}
ModelProcess::ModelProcess()
    :modelId_(0),
    loadFlag_(false),
    modelDesc_(nullptr),
    input_(nullptr),
    output_(nullptr),
    numInputs_(0),
    numOutputs_(0)
{
    str2aclAippInputFormat["YUV420SP_U8"] = ACL_YUV420SP_U8;
    str2aclAippInputFormat["XRGB8888_U8"] = ACL_XRGB8888_U8;
    str2aclAippInputFormat["RGB888_U8"] = ACL_RGB888_U8;
    str2aclAippInputFormat["YUV400_U8"] = ACL_YUV400_U8;
}

ModelProcess::~ModelProcess()
{
    Unload();
    DestroyDesc();
    DestroyInput(true);
    DestroyOutput(true);
}

Result ModelProcess::LoadModelFromFile(const string& modelPath)
{
    if (loadFlag_) {
        ERROR_LOG("has already loaded a model");
        return FAILED;
    }
    struct timeval start = { 0 };
    struct timeval end = { 0 };
    gettimeofday(&start, nullptr);
    aclError ret = aclmdlLoadFromFile(modelPath.c_str(), &modelId_);
    gettimeofday(&end, nullptr);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("load model from file failed, model file is %s", modelPath.c_str());
        return FAILED;
    }
    float time_cost = 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000.000;
    DEBUG_LOG("model aclmdlLoadFromFile cost : %f (ms)", time_cost);
    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath.c_str());
    return SUCCESS;
}

Result ModelProcess::CreateDesc()
{
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("get model description failed");
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

Result ModelProcess::GetDynamicGearCount(size_t &dymGearCount)
{
    aclError ret = aclmdlGetInputDynamicGearCount(modelDesc_, -1, &dymGearCount);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("get input dynamic gear count failed %d", ret);
        return FAILED;
    }

    DEBUG_LOG("get input dynamic gear count success");

    return SUCCESS;
}

Result ModelProcess::GetDynamicIndex(size_t &dymindex)
{
    aclError ret;

    const char *inputname = nullptr;
    bool dynamicIndex_exist = false;
    size_t numInputs = aclmdlGetNumInputs(modelDesc_);
    for (size_t i = 0; i < numInputs; i++) {
        inputname = aclmdlGetInputNameByIndex(modelDesc_, i);
        if (strcmp(inputname, ACL_DYNAMIC_TENSOR_NAME) == 0) {
            dynamicIndex_exist = true;
        }
    }
    if (dynamicIndex_exist == false) {
        g_dymindex = -1;
        return SUCCESS;
    }

    ret = aclmdlGetInputIndexByName(modelDesc_, ACL_DYNAMIC_TENSOR_NAME, &dymindex);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("get input index by name failed %d", ret);
        g_dymindex = -1;
        return FAILED;
    }
    DEBUG_LOG("get input index by name success");
    g_dymindex = dymindex;
    return SUCCESS;
}

Result ModelProcess::CheckDynamicShape(
    std::vector<std::string> dym_shape_tmp,
    std::map<string, std::vector<int64_t>> &dym_shape_map,
    std::vector<int64_t> &dims_num
)
{
    const char *inputname = nullptr;
    vector<const char *> inputnames;
    string name;
    string shape_str;
    size_t numInputs = aclmdlGetNumInputs(modelDesc_);
    int64_t num_tmp = 0;
    if (numInputs != dym_shape_tmp.size()) {
        ERROR_LOG("om has %zu input, but dymShape parametet give %zu", numInputs, dym_shape_tmp.size());
        return FAILED;
    }

    for (size_t i = 0; i < numInputs; i++) {
        inputname = aclmdlGetInputNameByIndex(modelDesc_, i);
        if (inputname == nullptr) {
            ERROR_LOG("get input name failed, index = %zu.", i);
            return FAILED;
        }
        inputnames.push_back(inputname);
    }
    for (size_t i = 0; i < dym_shape_tmp.size(); ++i) {
        istringstream block(dym_shape_tmp[i]);
        string cell;
        size_t index = 0;
        vector<string> shape_tmp;
        while (getline(block, cell, ':')) {
            if (index == 0) {
                name = cell;
            } else if (index == 1) {
                shape_str = cell;
            }
            index += 1;
        }
        Utils::SplitStringWithPunctuation(shape_str, shape_tmp, ',');
        size_t shape_tmp_size = shape_tmp.size();
        vector<int64_t> shape_array_tmp;

        dims_num.push_back(shape_tmp_size);
        for (size_t index = 0; index < shape_tmp_size; ++index) {
            num_tmp = atoi(shape_tmp[index].c_str());
            shape_array_tmp.push_back(num_tmp);
        }
        dym_shape_map[name] = shape_array_tmp;
    }
    for (size_t i = 0; i < inputnames.size(); ++i) {
        if (dym_shape_map.count(inputnames[i]) <= 0) {
            ERROR_LOG("the dymShape parameter set error, please check input name");
            return FAILED;
        }
    }
    DEBUG_LOG("check Dynamic Shape success");
    return SUCCESS;
}

Result ModelProcess::SetDynamicShape(
    std::map<std::string, std::vector<int64_t>> dym_shape_map,
    std::vector<int64_t> &dims_num
)
{
    aclError ret;
    const char *name;
    size_t  input_num = dym_shape_map.size();
    aclTensorDesc *inputDesc;
    for (size_t i = 0; i < input_num; i++) {
        name = aclmdlGetInputNameByIndex(modelDesc_, i);
        int64_t arr[dym_shape_map[name].size()];
        std::copy(dym_shape_map[name].begin(), dym_shape_map[name].end(), arr);
	    inputDesc = aclCreateTensorDesc(ACL_FLOAT, dims_num[i], arr, ACL_FORMAT_NCHW);
        ret = aclmdlSetDatasetTensorDesc(input_, inputDesc, i);
        if (ret != ACL_SUCCESS) {
            cout << aclGetRecentErrMsg() << endl;
            ERROR_LOG("aclmdlSetDatasetTensorDesc failed %d", ret);
            return FAILED;
        }
    }
    DEBUG_LOG("set Dynamic shape success");
    return SUCCESS;
}

Result ModelProcess::GetMaxDynamicHWSize(uint64_t &outsize)
{
    aclError ret;
    aclmdlHW dynamicHW;
    uint64_t maxDynamicHWSize = 0;
    ret = aclmdlGetDynamicHW(modelDesc_, -1, &dynamicHW);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("get DynamicHW failed");
        return FAILED;
    }

    if (dynamicHW.hwCount <= 0) {
        ERROR_LOG("the dynamic_image_size parameter is not specified for model conversion");
        return FAILED;
    }
    for (size_t i = 0; i < dynamicHW.hwCount; i++) {
        if (maxDynamicHWSize < (dynamicHW.hw[i][0] * dynamicHW.hw[i][1])) {
            maxDynamicHWSize = dynamicHW.hw[i][0] * dynamicHW.hw[i][1];
        }
    }
    outsize = maxDynamicHWSize;
    return SUCCESS;
}

Result ModelProcess::CheckDynamicHWSize(pair<int, int> dynamicPair, bool &is_dymHW)
{
    aclmdlHW dynamicHW;
    aclError ret;
    bool if_same = false;
    ret = aclmdlGetDynamicHW(modelDesc_, -1, &dynamicHW);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("get DynamicHW failed");
        return FAILED;
    }
    if (dynamicHW.hwCount > 0) {
        stringstream dynamicRange;
        for (size_t i = 0; i < dynamicHW.hwCount; i++) {
            if ((size_t)dynamicPair.first == dynamicHW.hw[i][0] and (size_t)dynamicPair.second == dynamicHW.hw[i][1]) {
                if_same = true;
                break;
            }
        }
        if (! if_same) {
            ERROR_LOG("the dymHW parameter is not correct");
            return FAILED;
        }
        is_dymHW = true;
    } else {
        ERROR_LOG("the dynamic_image_size parameter is not specified for model conversion");
        return FAILED;
    }
    INFO_LOG("check dynamic image size success.");
    return SUCCESS;
}

Result ModelProcess::SetDynamicHW(std::pair<uint64_t, uint64_t > dynamicPair)
{
    aclError ret = aclmdlSetDynamicHWSize(modelId_, input_, g_dymindex, dynamicPair.first, dynamicPair.second);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetDynamicHWSize failed %d", ret);
        return FAILED;
    }
    DEBUG_LOG("set Dynamic HW success");
    return SUCCESS;
}

Result ModelProcess::CheckDynamicBatchSize(uint64_t dymbatch, bool &is_dymbatch)
{
    aclmdlBatch batch_info;
    aclError ret;
    bool if_same = false;
    ret = aclmdlGetDynamicBatch(modelDesc_, &batch_info);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("get DynamicBatch failed");
        return FAILED;
    }
    if (batch_info.batchCount > 0) {
        for (size_t i = 0; i < batch_info.batchCount; i++) {
            if (dymbatch == batch_info.batch[i]) {
                if_same = true;
                break;
            }
        }
        if (!if_same) {
            ERROR_LOG("the dymBatch parameter is not correct");
            GetDymBatchInfo();
            return FAILED;
        }
        is_dymbatch = true;
    } else {
        ERROR_LOG("the dynamic_batch_size parameter is not specified for model conversion");
        return FAILED;
    }
    INFO_LOG("check dynamic batch success");
    return SUCCESS;
}

Result ModelProcess::SetDynamicBatchSize(uint64_t batchSize)
{
    aclError ret = aclmdlSetDynamicBatchSize(modelId_, input_, g_dymindex, batchSize);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetDynamicBatchSize failed %d", ret);
        return FAILED;
    }
    DEBUG_LOG("set dynamic batch size success");
    return SUCCESS;
}

Result ModelProcess::GetMaxBatchSize(uint64_t &maxBatchSize)
{
    aclmdlBatch batch_info;
    aclError ret = aclmdlGetDynamicBatch(modelDesc_, &batch_info);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("get DynamicBatch failed");
        return FAILED;
    }
    if (batch_info.batchCount > 0) {
        for (size_t i = 0; i < batch_info.batchCount; i++) {
            if (maxBatchSize < batch_info.batch[i]) {
                maxBatchSize = batch_info.batch[i];
            }
        }
    }
    DEBUG_LOG("get max dynamic batch size success");
    return SUCCESS;
}

Result ModelProcess::GetCurOutputDimsMul(size_t index, vector<int64_t>& curOutputDimsMul)
{
    aclError ret;
    aclmdlIODims ioDims;
    int64_t tmp_dim = 1;
    ret = aclmdlGetCurOutputDims(modelDesc_, index, &ioDims);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        WARN_LOG("aclmdlGetCurOutputDims failed ret[%d], maybe the modle has dynamic shape", ret);
        return FAILED;
    }
    for (size_t i = 1; i < ioDims.dimCount; i++) {
        tmp_dim *= ioDims.dims[ioDims.dimCount - i];
        curOutputDimsMul.push_back(tmp_dim);
    }
    return SUCCESS;
}


Result ModelProcess::CheckDynamicDims(vector<string> dym_dims, size_t gearCount, aclmdlIODims *dims)
{
    aclmdlGetInputDynamicDims(modelDesc_, -1, dims, gearCount);
    bool if_same = false;
    for (size_t i = 0; i < gearCount; i++) {
        if ((size_t)dym_dims.size() != dims[i].dimCount) {
            ERROR_LOG("the dymDims parameter is not correct i:%zu dysize:%zu dimcount:%zu",
                i, dym_dims.size(), dims[i].dimCount);
            GetDimInfo(gearCount, dims);
            return FAILED;
        }
        for (size_t j = 0; j < dims[i].dimCount; j++) {
            if (dims[i].dims[j] != atoi(dym_dims[j].c_str())) {
                break;
            }
            if (j == dims[i].dimCount - 1) {
                if_same = true;
            }
        }
    }

    if (!if_same) {
        ERROR_LOG("the dynamic_dims parameter is not correct");
        GetDimInfo(gearCount, dims);
        return FAILED;
    }
    DEBUG_LOG("check dynamic dims success");
    return SUCCESS;
}

Result ModelProcess::SetDynamicDims(vector<string> dym_dims)
{
    aclmdlIODims dims;
    dims.dimCount = dym_dims.size();
    for (size_t i = 0; i < dims.dimCount; i++) {
        dims.dims[i] = atoi(dym_dims[i].c_str());
    }

    aclError ret = aclmdlSetInputDynamicDims(modelId_, input_, g_dymindex, &dims);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetInputDynamicDims failed %d", ret);
        return FAILED;
    }
    DEBUG_LOG("set dynamic dims success");
    return SUCCESS;
}

void ModelProcess::GetDymBatchInfo()
{
    aclmdlBatch batch_info;
    aclmdlGetDynamicBatch(modelDesc_, &batch_info);
    stringstream ss;
    ss << "model has dynamic batch size:{";
    for (size_t i = 0; i < batch_info.batchCount; i++) {
        ss << "[" << batch_info.batch[i] << "]";
    }
    ss << "}, please set correct dynamic batch size";
    ERROR_LOG("%s", ss.str().c_str());
}

void ModelProcess::GetDymHWInfo()
{
    aclmdlHW  hw_info;
    aclmdlGetDynamicHW(modelDesc_, -1, &hw_info);
    stringstream ss;

    ERROR_LOG("model has %zu gear of HW", hw_info.hwCount);
    for (size_t i = 0; i < hw_info.hwCount; i++) {
        ss << "[" << hw_info.hw[i] << "]";
    }
    ss << "}, please set correct dynamic batch size";
    ERROR_LOG("%s", ss.str().c_str());
}

void ModelProcess::GetDimInfo(size_t gearCount, aclmdlIODims *dims)
{
    aclmdlGetInputDynamicDims(modelDesc_, -1, dims, gearCount);

    for (size_t i = 0; i < gearCount; i++) {
        if (i == 0) {
            ERROR_LOG("model has %zu gear of dims", gearCount);
        }
        stringstream ss;
        ss << "dims[" << i << "]:";
        for (size_t j = 0; j < dims[i].dimCount; j++) {
            ss << "[" << dims[i].dims[j] << "]";
        }
        ERROR_LOG("%s", ss.str().c_str());
    }
}

void ModelProcess::model_description(
    aclError ret, size_t& numInputs,
    size_t& numOutputs, aclmdlIODims& dimsInput,
    aclmdlIODims& dimsOutput
)
{
    for (size_t i = 0; i < numInputs; i++) {
        DEBUG_LOG("the size of %zu input: %zu", i, aclmdlGetInputSizeByIndex(modelDesc_, i));
        ret = aclmdlGetInputDims(modelDesc_, i, &dimsInput);
        DEBUG_LOG("the dims of %zu input:", i);
        for (size_t j = 0; j < dimsInput.dimCount; j++) {
            cout << dimsInput.dims[j] << " ";
        }
        cout << endl;
        DEBUG_LOG("the name of %zu input: %s", i, aclmdlGetInputNameByIndex(modelDesc_, i));
        DEBUG_LOG("the Format of %zu input: %u", i, aclmdlGetInputFormat(modelDesc_, i));
        DEBUG_LOG("the DataType of %zu input: %u", i, aclmdlGetInputDataType(modelDesc_, i));
    }
    for (size_t i = 0; i < numOutputs; i++) {
        DEBUG_LOG("the size of %zu output: %zu", i, aclmdlGetOutputSizeByIndex(modelDesc_, i));
        ret = aclmdlGetOutputDims(modelDesc_, i, &dimsOutput);
        DEBUG_LOG("the dims of %zu output:", i);
        for (size_t j = 0; j < dimsOutput.dimCount; j++) {
            cout << dimsOutput.dims[j] << " ";
        }
        cout << endl;

        DEBUG_LOG("the name of %zu output: %s", i, aclmdlGetOutputNameByIndex(modelDesc_, i));
        DEBUG_LOG("the Format of %zu output: %u", i, aclmdlGetOutputFormat(modelDesc_, i));
        DEBUG_LOG("the DataType of %zu output: %u", i, aclmdlGetOutputDataType(modelDesc_, i));
    }
    return;
}

Result ModelProcess::PrintDesc()
{
    aclError ret;
    DEBUG_LOG("start print model description");
    size_t numInputs = aclmdlGetNumInputs(modelDesc_);
    size_t numOutputs = aclmdlGetNumOutputs(modelDesc_);
    DEBUG_LOG("NumInputs: %zu", numInputs);
    DEBUG_LOG("NumOutputs: %zu", numOutputs);

    aclmdlIODims dimsInput;
    aclmdlIODims dimsOutput;
    model_description(ret, numInputs, numOutputs, dimsInput, dimsOutput);
    aclmdlBatch batch_info;
    ret = aclmdlGetDynamicBatch(modelDesc_, &batch_info);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("get DynamicBatch failed");
        return FAILED;
    }
    if (batch_info.batchCount != 0) {
        DEBUG_LOG("DynamicBatch:");
        for (size_t i = 0; i < batch_info.batchCount; i++) {
            cout << batch_info.batch[i] << " ";
        }
        cout << endl;
    }
    aclmdlHW dynamicHW;
    ret = aclmdlGetDynamicHW(modelDesc_, -1, &dynamicHW);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        modelDesc_ = nullptr;
        return FAILED;
    }
    if (dynamicHW.hwCount != 0) {
        DEBUG_LOG("DynamicHW:");
        for (size_t i = 0; i < dynamicHW.hwCount; i++) {
            cout << dynamicHW.hw[i][0] << "," <<dynamicHW.hw[i][1] << " ";
        }
        cout << endl;
    }
    DEBUG_LOG("end print model description");
    return SUCCESS;
}

void ModelProcess::DestroyDesc()
{
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

Result ModelProcess::CreateDymInput(size_t index)
{
    if (input_ == nullptr) {
        input_ = aclmdlCreateDataset();
        if (input_ == nullptr) {
            ERROR_LOG("can't create dataset, create input failed");
            return FAILED;
        }
    }
    size_t buffer_size = aclmdlGetInputSizeByIndex(modelDesc_, index);
    void* inBufferDev = nullptr;
    aclError ret = aclrtMalloc(&inBufferDev, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("malloc device buffer failed. size is %zu", buffer_size);
        return FAILED;
    }
    aclDataBuffer* inputData = aclCreateDataBuffer(inBufferDev, buffer_size);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        aclrtFree(inBufferDev);
        inBufferDev = nullptr;
        return FAILED;
    }
    ret = aclmdlAddDatasetBuffer(input_, inputData);
    DEBUG_LOG("add input_ at CreateDymInput +1");
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("add input dataset buffer failed");
        aclrtFree(inBufferDev);
        inBufferDev = nullptr;
        aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::UpdateInputsV1(const std::vector<int> &inOutRelation)
{
    if (input_ == nullptr || output_ == nullptr) {
        if (input_ == nullptr) {ERROR_LOG("can't find inputdatas");}
        if (output_ == nullptr) {ERROR_LOG("can't find outputdatas");}
        return FAILED;
    }
    size_t inputsNum = aclmdlGetDatasetNumBuffers(input_);
    size_t outputsNum = aclmdlGetDatasetNumBuffers(output_);
    if (inputsNum != inOutRelation.size()) {
        ERROR_LOG("wrong inOutRelation size");
        return FAILED;
    }

    for (size_t i = 0; i < inputsNum; ++i) {
        aclError ret;
        if (inOutRelation[i] < 0) {
            continue;
        } else if (inOutRelation[i] < outputsNum) {
            aclDataBuffer* tmpInputData = aclmdlGetDatasetBuffer(input_, i);
            aclDataBuffer* tmpOutputData = aclmdlGetDatasetBuffer(output_, inOutRelation[i]);
            if (aclGetDataBufferSizeV2(tmpInputData) != aclGetDataBufferSizeV2(tmpOutputData)) {
                ERROR_LOG("inputSize_current and outputSize_last not matched");
                return FAILED;
            }
            size_t tensorSize = aclGetDataBufferSizeV2(tmpOutputData);
            void* inBuffer = aclGetDataBufferAddr(tmpInputData);
            void* outBuffer = aclGetDataBufferAddr(tmpOutputData);
            ret = aclUpdateDataBuffer(tmpInputData, outBuffer, tensorSize);
            if (ret != ACL_SUCCESS) {
                cout << aclGetRecentErrMsg() << endl;
                ERROR_LOG("new input buffer aclrtMemcpy from last output failed. size is %zu", tensorSize);
                return FAILED;
            }
            if (!reuseOutput_) {
                (void)aclrtFree(inBuffer);
            }
        } else {
            ERROR_LOG("find outputdata index out of range");
            return FAILED;
        }
    }
    if (!reuseOutput_) {
        reuseOutput_ = true;
    }
    return SUCCESS;
}

Result ModelProcess::UpdateInputsV2(const std::vector<int> &inOutRelation)
{
    if (input_ == nullptr || output_ == nullptr) {
        if (input_ == nullptr) {ERROR_LOG("can't find inputdatas");}
        if (output_ == nullptr) {ERROR_LOG("can't find outputdatas");}
        return FAILED;
    }
    size_t inputsNum = aclmdlGetDatasetNumBuffers(input_);
    size_t outputsNum = aclmdlGetDatasetNumBuffers(output_);
    if (inputsNum != inOutRelation.size()) {
        ERROR_LOG("wrong inOutRelation size");
        return FAILED;
    }

    for (size_t i = 0; i < inputsNum; ++i) {
        aclError ret;
        if (inOutRelation[i] < 0) {
            continue;
        } else if (inOutRelation[i] < outputsNum) {
            aclDataBuffer* tmpInputData = aclmdlGetDatasetBuffer(input_, i);
            aclDataBuffer* tmpOutputData = aclmdlGetDatasetBuffer(output_, inOutRelation[i]);
            if (aclGetDataBufferSizeV2(tmpInputData) != aclGetDataBufferSizeV2(tmpOutputData)) {
                ERROR_LOG("inputSize_current and outputSize_last not matched");
                return FAILED;
            }
            size_t tensorSize = aclGetDataBufferSizeV2(tmpOutputData);
            void* lastBuffer = aclGetDataBufferAddr(tmpInputData);
            void* lastOutBuffer = aclGetDataBufferAddr(tmpOutputData);
            ret = aclrtMemcpy(lastBuffer, tensorSize, lastOutBuffer, tensorSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                cout << aclGetRecentErrMsg() << endl;
                ERROR_LOG("new input buffer aclrtMemcpy from last output failed. size is %zu", tensorSize);
                return FAILED;
            }
        } else {
            ERROR_LOG("find outputdata index out of range");
            return FAILED;
        }
    }

    return SUCCESS;
}

Result ModelProcess::CreateInput(void* inputDataBuffer, size_t bufferSize)
{
    if (input_ == nullptr) {
        input_ = aclmdlCreateDataset();
        if (input_ == nullptr) {
            ERROR_LOG("can't create dataset, create input failed");
            return FAILED;
        }
    }

    aclDataBuffer* inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    DEBUG_LOG("add input_ at CreateInput +1");
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("add input dataset buffer failed");
        aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::check_ret(aclError ret, size_t buffer_size_zero)
{
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("malloc device buffer failed. size is %zu", buffer_size_zero);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::check_create_buffer(aclDataBuffer* inputData, void* inBufferDev)
{
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        aclrtFree(inBufferDev);
        inBufferDev = nullptr;
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::check_add_buffer(aclError ret, void* inBufferDev, aclDataBuffer* inputData)
{
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("add input dataset buffer failed");
        aclrtFree(inBufferDev);
        inBufferDev = nullptr;
        aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateZeroInput()
{
    if (input_ == nullptr) {
        input_ = aclmdlCreateDataset();
        if (input_ == nullptr) {
            ERROR_LOG("can't create dataset, create input failed");
            return FAILED;
        }
    }
    aclError ret;
    numInputs_ = aclmdlGetNumInputs(modelDesc_);
    for (size_t i = 0; i < numInputs_; i++) {
        const char *name = aclmdlGetInputNameByIndex(modelDesc_, i);
        if (name == nullptr) {
            ERROR_LOG("get input name failed, index = %zu.", i);
            return FAILED;
        }

        size_t buffer_size_zero = aclmdlGetInputSizeByIndex(modelDesc_, i);
        void* inBufferDev = nullptr;

        ret = aclrtMalloc(&inBufferDev, buffer_size_zero, ACL_MEM_MALLOC_HUGE_FIRST);
        if (check_ret(ret, buffer_size_zero) == FAILED) {
            return FAILED;
        }
        if (strcmp(name, ACL_DYNAMIC_TENSOR_NAME) != 0) {
            ret = aclrtMemset(inBufferDev, buffer_size_zero, 0, buffer_size_zero);
            if (ret != ACL_SUCCESS) {
                cout << aclGetRecentErrMsg() << endl;
                ERROR_LOG("memory set failed");
                aclrtFree(inBufferDev);
                inBufferDev = nullptr;
                return FAILED;
            }
        }

        aclDataBuffer* inputData = aclCreateDataBuffer(inBufferDev, buffer_size_zero);
        if (check_create_buffer(inputData, inBufferDev) == FAILED) {
            return FAILED;
        }
        ret = aclmdlAddDatasetBuffer(input_, inputData);
        DEBUG_LOG("add input_ at CreateZeroInput +1");
        if (check_add_buffer(ret, inBufferDev, inputData) == FAILED) {
            return FAILED;
        }
    }
    return SUCCESS;
}

void ModelProcess::DestroyInput(bool free_memory_flag = true)
{
    if (input_ == nullptr) {
        return;
    }

    size_t bufNum = aclmdlGetDatasetNumBuffers(input_);
    for (size_t i = 0; i < bufNum; ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        if (dataBuffer == nullptr) {
            continue;
        }
        void *data = aclGetDataBufferAddr(dataBuffer);
        if (data == nullptr) {
            (void)aclDestroyDataBuffer(dataBuffer);
            continue;
        }
        if (free_memory_flag == true) {
            (void)aclrtFree(data);
            data = nullptr;
        }
        (void)aclDestroyDataBuffer(dataBuffer);
        dataBuffer = nullptr;
    }
    (void)aclmdlDestroyDataset(input_);
    input_ = nullptr;
    DEBUG_LOG("destroy model input success");
}

Result ModelProcess::CreateOutput()
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create ouput failed");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputNum = aclmdlGetNumOutputs(modelDesc_);
    if ((g_output_size.empty() == false)  && (outputNum != g_output_size.size())) {
        ERROR_LOG("om has %zu output, but outputSize parametet give %zu", outputNum, g_output_size.size());
        return FAILED;
    }

    for (size_t i = 0; i < outputNum; ++i) {
        size_t buffer_size = 0;
        if (g_output_size.empty() == false) {
            buffer_size = g_output_size[i];
        } else {
            buffer_size = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        }
        void* outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            cout << aclGetRecentErrMsg() << endl;
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", buffer_size);
            return FAILED;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        if (outputData == nullptr) {
            cout << aclGetRecentErrMsg() << endl;
            ERROR_LOG("can't create data buffer, create output failed");
            aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            cout << aclGetRecentErrMsg() << endl;
            ERROR_LOG("can't add data buffer, create output failed");
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    INFO_LOG("create model output success");
    return SUCCESS;
}

void ModelProcess::print_float_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(float); i++) {
        float out = *((float*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_aclFloat16_info(
    size_t len, std::ofstream& outstr,
    void* outData, vector<int64_t> curOutputDimsMul
)
{
    aclFloat16 *out_fp16 = reinterpret_cast<aclFloat16*>(outData);
    float out = 0;
    for (size_t i = 1; i <= len / sizeof(aclFloat16); i++) {
        out = aclFloat16ToFloat(out_fp16[i-1]);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_int8_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(int8_t); i++) {
        int8_t out = *((int8_t*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
            outstr << "\n";
            break;
            }
        }
    }
    return;
}

void ModelProcess::print_int_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(int); i++) {
        int out = *((int*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_uint8_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(uint8_t); i++) {
        uint8_t out = *((uint8_t*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_int16_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(int16_t); i++) {
        int16_t out = *((int16_t*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_uint16_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(uint16_t); i++) {
        uint16_t out = *((uint16_t*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_uint32_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(uint32_t); i++) {
        uint32_t out = *((uint32_t*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_int64_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(int64_t); i++) {
        int64_t out = *((int64_t*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_uint64_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(uint64_t); i++) {
        uint64_t out = *((uint64_t*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_double_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(double); i++) {
        double out = *((double*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_bool_info(size_t len, std::ofstream& outstr, void* outData, vector<int64_t> curOutputDimsMul)
{
    for (size_t i = 1; i <= len / sizeof(bool); i++) {
        int out = *((bool*)outData + i - 1);
        outstr << out << " ";
        vector<int64_t>::iterator it;
        for (it = curOutputDimsMul.begin(); it != curOutputDimsMul.end(); it++) {
            if ((i != 0) && (i % *it == 0)) {
                outstr << "\n";
                break;
            }
        }
    }
    return;
}

void ModelProcess::print_data_log(
    aclDataType datatype, size_t len, std::ofstream& outstr,
    void* outData, vector<int64_t> curOutputDimsMul
)
{
    switch (datatype) {
        case TYPE_FLOAT:
            print_float_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_ACLFLOAT16:
            print_aclFloat16_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_INT8_T:
            print_int8_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_INT:
            print_int_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_UINT8_T:
            print_uint8_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_INT16_T:
            print_int16_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_UINT16_T:
            print_uint16_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_UINT32_T:
            print_uint32_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_INT64_T:
            print_int64_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_UINT64_T:
            print_uint64_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_DOUBLE:
            print_double_info(len, outstr, outData, curOutputDimsMul);
            break;
        case TYPE_BOOL:
            print_bool_info(len, outstr, outData, curOutputDimsMul);
            break;
        default:
            printf("undefined data type!\n");
            break;
    }
    return;
}

Result ModelProcess::Free_Host_Try(aclError ret, void*& outHostData)
{
    if (!g_isDevice) {
        ret = aclrtFreeHost(outHostData);
        if (ret != ACL_SUCCESS) {
            cout << aclGetRecentErrMsg() << endl;
            ERROR_LOG("aclrtFreeHost failed, ret[%d]", ret);
            return FAILED;
        }
    }
    return SUCCESS;
}

void ModelProcess::print_error_log(aclError ret)
{
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclrtMemcpy failed, ret[%d]", ret);
    }
    return;
}

void ModelProcess::DestroyOutput(bool free_memory_flag = true)
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        if (free_memory_flag == true) {
            (void)aclrtFree(data);
        }
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
}

Result GetDescShape(const aclTensorDesc *desc, std::vector<int64_t>& shape)
{
    size_t dimNums = aclGetTensorDescNumDims(desc);
    if (dimNums == ACL_UNKNOWN_RANK) {
        WARN_LOG("GetDescDimsNum failed unknown rank");
        return FAILED;
    }
    aclError ret;
    for (size_t i = 0; i < dimNums; ++i) {
        int64_t dim;
        ret = aclGetTensorDescDimV2(desc, i, &dim);
        if (ret != ACL_SUCCESS) {
            WARN_LOG("GetDescDims i:%zu dimsNum:%zu failed ret:%d", i, dimNums, ret);
            return FAILED;
        }
        shape.push_back(dim);
    }
    return SUCCESS;
}

Result GetDescShapeStr(const aclTensorDesc *desc, std::string &shapestr)
{
    std::vector<int64_t> shape;
    Result result = GetDescShape(desc, shape);
    if (result != SUCCESS) {
        return FAILED;
    }

    for (size_t i = 0; i < shape.size(); i++) {
        if (i == 0) {
            shapestr +=  std::to_string(shape[i]);
        } else {
            shapestr += "x" + std::to_string(shape[i]);
        }
    }
    return SUCCESS;
}

Result SaveTensorMemoryToFile(const aclTensorDesc *desc, std::string &prefixName)
{
    aclError ret;
    aclFormat format = aclGetTensorDescFormat(desc);
    aclDataType dtype = aclGetTensorDescType(desc);
    std::string shapestr;
    if (GetDescShapeStr(desc, shapestr) != SUCCESS) {
        WARN_LOG("exception_cb get shape failed continue");
    }
    void *devaddr = aclGetTensorDescAddress(desc);
    size_t len = aclGetTensorDescSize(desc);
    if (devaddr == nullptr || len == 0) {
        WARN_LOG("exception_cb get failed addr:%p len:%zu", devaddr, len);
        return FAILED;
    }
    void* hostaddr = nullptr;
    ret = aclrtMallocHost(&hostaddr, len);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        WARN_LOG("exception_cb MallocHost failed len:%zu ret:%d", len, ret);
        return FAILED;
    }
    ret = aclrtMemcpy(hostaddr, len, devaddr, len, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        WARN_LOG("exception_cb aclMemcpy failed ret:%d hostaddr:%p devaddr:%p len:%zu",
            ret, hostaddr, devaddr, len);
        return FAILED;
    }
    std::string fileName = prefixName + "_format_" + std::to_string(format) +
        "_dtype_" + std::to_string(dtype) + "_shape_" + shapestr + ".bin";
    INFO_LOG("exception_cb hostaddr:%p devaddr:%p len:%zu write to filename:%s",
             hostaddr, devaddr, len, fileName.c_str());
    ofstream outFile(fileName, ios::out | ios::binary);
    outFile.write((char*)hostaddr, len);
    return SUCCESS;
}

void callback(aclrtExceptionInfo *exceptionInfo)
{
    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(exceptionInfo);
    if (deviceId == 0xffffffff) {
        WARN_LOG("exception_cb get exception deviceId failed");
        return;
    }
    uint32_t streamId = aclrtGetStreamIdFromExceptionInfo(exceptionInfo);
    if (streamId == 0xffffffff) {
        WARN_LOG("exception_cb get exception streamId failed");
        return;
    }
    uint32_t taskId = aclrtGetTaskIdFromExceptionInfo(exceptionInfo);
    if (taskId == 0xffffffff) {
        WARN_LOG("exception_cb get exception taskId failed");
        return;
    }

    char opName[256];
    aclTensorDesc *inputDesc = nullptr;
    aclTensorDesc *outputDesc = nullptr;
    size_t inputCnt = 0;
    size_t outputCnt = 0;
    aclError ret = aclmdlCreateAndGetOpDesc(deviceId, streamId, taskId, opName, 256, \
                                            &inputDesc, &inputCnt, &outputDesc, &outputCnt);
    if (ret != ACL_SUCCESS) {
        WARN_LOG("exception_cb deviceId:%u streamId:%u taskId:%u failed:%d", deviceId, streamId, taskId, ret);
        return;
    }

    static int index = 0;
    INFO_LOG("exception_cb streamId:%u taskId:%u deviceId: %u opName:%s inputCnt:%zu outputCnt:%zu",
        streamId, taskId, deviceId, opName, inputCnt, outputCnt);
    for (size_t i = 0; i < inputCnt; ++i) {
        const aclTensorDesc *desc = aclGetTensorDescByIndex(inputDesc, i);
        std::string prefix_filename = "exception_cb_index_" + std::to_string(index) + \
            "_input_" + std::to_string(i);
        if (SaveTensorMemoryToFile(desc, prefix_filename) != SUCCESS) {
            WARN_LOG("exception_cb input_%zu save failed", i);
            break;
        }
    }
    for (size_t i = 0; i < outputCnt; ++i) {
        const aclTensorDesc *desc = aclGetTensorDescByIndex(outputDesc, i);
        std::string prefix_filename = "exception_cb_index_" + std::to_string(index) + \
            "_output_" + std::to_string(i);
        if (SaveTensorMemoryToFile(desc, prefix_filename) != SUCCESS) {
            WARN_LOG("exception_cb input_%zu save failed", i);
            break;
        }
    }
    index++;
    aclDestroyTensorDesc(inputDesc);
    aclDestroyTensorDesc(outputDesc);
}

void ModelProcess::SetExceptionCallBack()
{
    aclrtSetExceptionInfoCallback(callback);
}

void ModelProcess::InitReuseOutput()
{
    reuseOutput_ = false;
}

Result ModelProcess::Execute()
{
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }

    DEBUG_LOG("model execute success");
    return SUCCESS;
}

void ModelProcess::Unload()
{
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
    loadFlag_ = false;
    INFO_LOG("unload model success, model Id is %u", modelId_);
}

Result ModelProcess::GetCurOutputShape(size_t index, bool is_dymshape, std::vector<int64_t>& shape)
{
    aclError ret;
    aclmdlIODims ioDims;
    // 对于动态shape场景，通过V2接口获取，其他通过V1接口获取
    if (is_dymshape == true) {
        aclTensorDesc *outputDesc = aclmdlGetDatasetTensorDesc(output_, index);
        size_t dimNums = aclGetTensorDescNumDims(outputDesc);
        if (dimNums == ACL_UNKNOWN_RANK) {
            return FAILED;
        } else {
            for (size_t i = 0; i < dimNums; ++i) {
                int64_t dim;
                ret = aclGetTensorDescDimV2(outputDesc, i, &dim);
                shape.push_back(dim);
            }
        }
    } else {
        ret = aclmdlGetCurOutputDims(modelDesc_, index, &ioDims);
        if (ret != ACL_SUCCESS) {
            DEBUG_LOG("aclmdlGetCurOutputDims get not success, maybe the modle has dynamic shape.ret=%d", ret);
            return FAILED;
        }
        for (size_t i = 0; i < ioDims.dimCount; i++) {
            shape.push_back(ioDims.dims[i]);
        }
    }
    return SUCCESS;
}

size_t ModelProcess::GetNumInputs()
{
    return aclmdlGetNumInputs(modelDesc_);
}

size_t ModelProcess::GetNumOutputs()
{
    return aclmdlGetNumOutputs(modelDesc_);
}

Result ModelProcess::GetInTensorDesc(
    size_t i, std::string& name, int& datatype,
    size_t& format, std::vector<int64_t>& shape, size_t& size
)
{
    name = aclmdlGetInputNameByIndex(modelDesc_, i);
    datatype = aclmdlGetInputDataType(modelDesc_, i);
    format = aclmdlGetInputFormat(modelDesc_, i);
    size = aclmdlGetInputSizeByIndex(modelDesc_, i);

    aclmdlIODims dimsInput;
    aclmdlGetInputDims(modelDesc_, i, &dimsInput);

    shape.clear();
    for (size_t j = 0; j < dimsInput.dimCount; j++) {
        shape.push_back(dimsInput.dims[j]);
    }
    return SUCCESS;
}

Result ModelProcess::GetOutTensorDesc(
    size_t i, std::string& name, int& datatype,
    size_t& format, std::vector<int64_t>& shape, size_t& size
)
{
    name = aclmdlGetOutputNameByIndex(modelDesc_, i);
    datatype = aclmdlGetOutputDataType(modelDesc_, i);
    format = aclmdlGetOutputFormat(modelDesc_, i);
    size = aclmdlGetOutputSizeByIndex(modelDesc_, i);

    aclmdlIODims dimsOutput;
    aclmdlGetOutputDims(modelDesc_, i, &dimsOutput);

    shape.clear();
    for (size_t j = 0; j < dimsOutput.dimCount; j++) {
        shape.push_back(dimsOutput.dims[j]);
    }
    return SUCCESS;
}

size_t ModelProcess::GetOutTensorLen(size_t i, bool is_dymshape)
{
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
    uint64_t maxBatchSize = 0;
    size_t len;
    GetMaxBatchSize(maxBatchSize);
    if (is_dymshape) {
	    aclTensorDesc *outputDesc = aclmdlGetDatasetTensorDesc(output_, i);
	    len = aclGetTensorDescSize(outputDesc);
    } else {
	    len = aclGetDataBufferSizeV2(dataBuffer);
    }
    return len;
}

Result ModelProcess::CreateOutput(void* outputBuffer, size_t bufferSize)
{
    if (output_ == nullptr) {
        output_ = aclmdlCreateDataset();
        if (output_ == nullptr) {
            ERROR_LOG("can't create dataset, create output failed");
            return FAILED;
        }
    }

    aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, bufferSize);
    if (outputData == nullptr) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("can't create data buffer, create output failed");
        aclrtFree(outputBuffer);
        return FAILED;
    }

    aclError ret = aclmdlAddDatasetBuffer(output_, outputData);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("add input dataset buffer failed");
        aclDestroyDataBuffer(outputData);
        outputData = nullptr;
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::FreeAIPP(aclmdlAIPP* aippParmsSet)
{
    aclError ret = aclmdlDestroyAIPP(aippParmsSet);
    if (ret != ACL_SUCCESS) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("free aclmdlAIPP failed");
        return FAILED;
    }
    aippParmsSet = nullptr;
    return SUCCESS;
}

int ModelProcess::CheckDymAIPPInputExist()
{
    /*
    模型有没有动态AIPP输入，用aclmdlGetAippType 函数找找，能找到说明模型没问题
    */
    size_t numInputs = aclmdlGetNumInputs(modelDesc_);
    std::vector<size_t> dataNeedDynamicAipp = {};
    DEBUG_LOG("Input nums: %d", int(numInputs));
    DEBUG_LOG("Model id: %u", modelId_);
    for (size_t index = 0; index < numInputs; ++index) {
        aclmdlInputAippType aippType;
        size_t dynamicAttachedDataIndex;
        aclError ret = aclmdlGetAippType(modelId_, index, &aippType, &dynamicAttachedDataIndex);
        if (ret != ACL_SUCCESS) {
            cout << aclGetRecentErrMsg() << endl;
            ERROR_LOG("aclmdlGetAippType failed");
            return -1;
        }
        if (aippType == ACL_DATA_WITH_DYNAMIC_AIPP) {
            dataNeedDynamicAipp.push_back(index);
        }
    }
    int aippNum = dataNeedDynamicAipp.size();
    return aippNum;
}

Result ModelProcess::GetAIPPIndexList(std::vector<size_t> &dataNeedDynamicAipp)
{
    // 获取标识动态AIPP输入的index
    // modelDesc_为aclmdlCreateDesc表示模型描述信息，根据1中加载成功的模型的ID，获取该模型的描述信息
    const char *inputName = nullptr;
    for (size_t index = 0; index < aclmdlGetNumInputs(modelDesc_); ++index) {
        inputName = aclmdlGetInputNameByIndex(modelDesc_, index);
        if (strcmp(inputName, ACL_DYNAMIC_AIPP_NAME) == 0) {
            dataNeedDynamicAipp.push_back(index);
            break;
        }
    }

    if (dataNeedDynamicAipp.size() == 0) {
        return FAILED;
    }
    INFO_LOG("GetAIPPIndex success");
    return SUCCESS;
}

Result ModelProcess::SetInputAIPP(size_t index, void* pAippDynamicSet)
{
    DEBUG_LOG("PREPARE aclmdlSetInputAIPP");
    aclError ret = aclmdlSetInputAIPP(modelId_, input_, index, (aclmdlAIPP *)pAippDynamicSet);
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetInputAIPP failed, index:%d ret %d", int(index), ret);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPSrcImageSize(std::shared_ptr<Base::DynamicAippConfig> dyAippCfg, aclmdlAIPP* aippDynamicSet)
{
    DEBUG_LOG("aclmdlSetAIPPSrcImageSize params: aippParmsSet: %p srcImageSizeW: %d srcImageSizeH: %d",
        aippDynamicSet, dyAippCfg->GetSrcImageSizeW(), dyAippCfg->GetSrcImageSizeH());
    aclError ret = aclmdlSetAIPPSrcImageSize(aippDynamicSet,
        dyAippCfg->GetSrcImageSizeW(), dyAippCfg->GetSrcImageSizeH());
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPSrcImageSize failed, w: %d, h: %d, ret: %d", dyAippCfg->GetSrcImageSizeW(),
            dyAippCfg->GetSrcImageSizeH(), ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPInputFormat(std::shared_ptr<Base::DynamicAippConfig> dyAippCfg, aclmdlAIPP* aippDynamicSet)
{
    DEBUG_LOG("aclmdlSetAIPPInputFormat, params: aippParmsSet: %p inputFormat: %s",
        aippDynamicSet, dyAippCfg->GetInputFormat().c_str());
    aclError ret = aclmdlSetAIPPInputFormat(aippDynamicSet, str2aclAippInputFormat[dyAippCfg->GetInputFormat()]);
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPInputFormat failed, ret %d", ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPCscParams(std::shared_ptr<Base::DynamicAippConfig> dyAippCfg, aclmdlAIPP* aippDynamicSet)
{
    DEBUG_LOG("aclmdlSetAIPPCscParams, params: aippParmsSet: %p csc_switch: %d cscMatrixR0C0: %d cscMatrixR0C1: %d \
        cscMatrixR0C2: %d cscMatrixR1C0: %d cscMatrixR1C1: %d cscMatrixR1C2: %d cscMatrixR2C0: %d cscMatrixR2C1: %d \
        cscMatrixR2C2: %d cscOutputBias0: %d cscOutputBias1: %d cscOutputBias2: %d cscInputBias0: %d \
        cscInputBias1: %d cscInputBias2: %d", aippDynamicSet, dyAippCfg->GetCscParams().cscSwitch,
        dyAippCfg->GetCscParams().cscMatrixR0C0, dyAippCfg->GetCscParams().cscMatrixR0C1,
        dyAippCfg->GetCscParams().cscMatrixR0C2, dyAippCfg->GetCscParams().cscMatrixR1C0,
        dyAippCfg->GetCscParams().cscMatrixR1C1, dyAippCfg->GetCscParams().cscMatrixR1C2,
        dyAippCfg->GetCscParams().cscMatrixR2C0, dyAippCfg->GetCscParams().cscMatrixR2C1,
        dyAippCfg->GetCscParams().cscMatrixR2C2, dyAippCfg->GetCscParams().cscOutputBias0,
        dyAippCfg->GetCscParams().cscOutputBias1, dyAippCfg->GetCscParams().cscOutputBias2,
        dyAippCfg->GetCscParams().cscInputBias0, dyAippCfg->GetCscParams().cscInputBias1,
        dyAippCfg->GetCscParams().cscInputBias2);
    aclError ret = aclmdlSetAIPPCscParams(aippDynamicSet, dyAippCfg->GetCscParams().cscSwitch,
        dyAippCfg->GetCscParams().cscMatrixR0C0, dyAippCfg->GetCscParams().cscMatrixR0C1,
        dyAippCfg->GetCscParams().cscMatrixR0C2, dyAippCfg->GetCscParams().cscMatrixR1C0,
        dyAippCfg->GetCscParams().cscMatrixR1C1, dyAippCfg->GetCscParams().cscMatrixR1C2,
        dyAippCfg->GetCscParams().cscMatrixR2C0, dyAippCfg->GetCscParams().cscMatrixR2C1,
        dyAippCfg->GetCscParams().cscMatrixR2C2, dyAippCfg->GetCscParams().cscOutputBias0,
        dyAippCfg->GetCscParams().cscOutputBias1, dyAippCfg->GetCscParams().cscOutputBias2,
        dyAippCfg->GetCscParams().cscInputBias0, dyAippCfg->GetCscParams().cscInputBias1,
        dyAippCfg->GetCscParams().cscInputBias2);
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPCscParams failed, ret %d", ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPRbuvSwapSwitch(
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    aclmdlAIPP* aippDynamicSet
)
{
    DEBUG_LOG("aclmdlSetAIPPRbuvSwapSwitch paras: aippParmsSet: %p rbuvSwapSwitch: %d",
        aippDynamicSet, dyAippCfg->GetRbuvSwapSwitch());
    aclError ret = aclmdlSetAIPPRbuvSwapSwitch(aippDynamicSet, dyAippCfg->GetRbuvSwapSwitch());
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPRbuvSwapSwitch failed rbuvSwap:%d aippset:%p ret %d",
            dyAippCfg->GetRbuvSwapSwitch(), aippDynamicSet, ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPAxSwapSwitch(
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    aclmdlAIPP* aippDynamicSet
)
{
    DEBUG_LOG("aclmdlSetAIPPAxSwapSwitch paras: aippDynamicSet: %p axSwapSwitch: %d",
        aippDynamicSet, dyAippCfg->GetAxSwapSwitch());
    aclError ret = aclmdlSetAIPPAxSwapSwitch(aippDynamicSet, dyAippCfg->GetAxSwapSwitch());
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPAxSwapSwitch failed, ret %d", ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPDtcPixelMean(
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    aclmdlAIPP* aippDynamicSet, size_t batchIndex
)
{
    aclError ret = ACL_ERROR_NONE;
    int dtcPixelMeanIndex = GetDynamicAippParaByBatch(batchIndex, dyAippCfg, "dtcPixelMean");
    if (dtcPixelMeanIndex >= 0) {
        DEBUG_LOG("aclmdlSetAIPPDtcPixelMean params: aippDynamicSet: %p dtcPixelMeanChn0: %d dtcPixelMeanChn1: %d\
            dtcPixelMeanChn2: %d dtcPixelMeanChn3: %d batchIndex: %d", aippDynamicSet,
            dyAippCfg->GetDtcPixelMean()[dtcPixelMeanIndex].dtcPixelMeanChn0,
            dyAippCfg->GetDtcPixelMean()[dtcPixelMeanIndex].dtcPixelMeanChn1,
            dyAippCfg->GetDtcPixelMean()[dtcPixelMeanIndex].dtcPixelMeanChn2,
            dyAippCfg->GetDtcPixelMean()[dtcPixelMeanIndex].dtcPixelMeanChn3, int(batchIndex));
        ret = aclmdlSetAIPPDtcPixelMean(aippDynamicSet,
            dyAippCfg->GetDtcPixelMean()[dtcPixelMeanIndex].dtcPixelMeanChn0,
            dyAippCfg->GetDtcPixelMean()[dtcPixelMeanIndex].dtcPixelMeanChn1,
            dyAippCfg->GetDtcPixelMean()[dtcPixelMeanIndex].dtcPixelMeanChn2,
            dyAippCfg->GetDtcPixelMean()[dtcPixelMeanIndex].dtcPixelMeanChn3, batchIndex);
    } else {
        DEBUG_LOG("aclmdlSetAIPPDtcPixelMean params: aippDynamicSet: %p dtcPixelMeanChn0: %d dtcPixelMeanChn1: %d\
            dtcPixelMeanChn2: %d dtcPixelMeanChn3: %d batchIndex: %d", aippDynamicSet, 0, 0, 0, 0, int(batchIndex));
        ret = aclmdlSetAIPPDtcPixelMean(aippDynamicSet, 0, 0, 0, 0, batchIndex);
    }
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPDtcPixelMean failed, ret %d", ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPDtcPixelMin(
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    aclmdlAIPP* aippDynamicSet, size_t batchIndex
)
{
    aclError ret = ACL_ERROR_NONE;
    int dtcPixelMinIndex = GetDynamicAippParaByBatch(batchIndex, dyAippCfg, "dtcPixelMin");
    if (dtcPixelMinIndex >= 0) {
        DEBUG_LOG("aclmdlSetAIPPDtcPixelMin params: %p dtcPixelMinChn0: %f dtcPixelMinChn1: %f dtcPixelMinChn2: %f \
            dtcPixelMinChn3 %f batchIndex: %d", aippDynamicSet,
            dyAippCfg->GetDtcPixelMin()[dtcPixelMinIndex].dtcPixelMinChn0,
            dyAippCfg->GetDtcPixelMin()[dtcPixelMinIndex].dtcPixelMinChn1,
            dyAippCfg->GetDtcPixelMin()[dtcPixelMinIndex].dtcPixelMinChn2,
            dyAippCfg->GetDtcPixelMin()[dtcPixelMinIndex].dtcPixelMinChn3, int(batchIndex));
        ret = aclmdlSetAIPPDtcPixelMin(aippDynamicSet,
            dyAippCfg->GetDtcPixelMin()[dtcPixelMinIndex].dtcPixelMinChn0,
            dyAippCfg->GetDtcPixelMin()[dtcPixelMinIndex].dtcPixelMinChn1,
            dyAippCfg->GetDtcPixelMin()[dtcPixelMinIndex].dtcPixelMinChn2,
            dyAippCfg->GetDtcPixelMin()[dtcPixelMinIndex].dtcPixelMinChn3, batchIndex);
    } else {
        DEBUG_LOG("aclmdlSetAIPPDtcPixelMin params: %p dtcPixelMinChn0: %f dtcPixelMinChn1: %f dtcPixelMinChn2: %f \
            dtcPixelMinChn3 %f batchIndex: %d", aippDynamicSet, 0.0, 0.0, 0.0, 0.0, int(batchIndex));
        ret = aclmdlSetAIPPDtcPixelMin(aippDynamicSet, 0.0, 0.0, 0.0, 0.0, batchIndex);
    }
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPDtcPixelMin failed, ret %d", ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPPixelVarReci(
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    aclmdlAIPP* aippDynamicSet, size_t batchIndex
)
{
    aclError ret = ACL_ERROR_NONE;
    int pixelVarReciIndex = GetDynamicAippParaByBatch(batchIndex, dyAippCfg, "pixelVarReci");
    if (pixelVarReciIndex >= 0) {
        DEBUG_LOG("aclmdlSetAIPPPixelVarReci params: aippDynamicSet: %p dtcPixelVarReciChn0: %f dtcPixelVarReciChn1: \
            %f dtcPixelVarReciChn2: %f dtcPixelVarReciChn3: %f batchIndex: %d", aippDynamicSet,
            dyAippCfg->GetPixelVarReci()[pixelVarReciIndex].dtcPixelVarReciChn0,
            dyAippCfg->GetPixelVarReci()[pixelVarReciIndex].dtcPixelVarReciChn1,
            dyAippCfg->GetPixelVarReci()[pixelVarReciIndex].dtcPixelVarReciChn2,
            dyAippCfg->GetPixelVarReci()[pixelVarReciIndex].dtcPixelVarReciChn3, int(batchIndex));
        ret = aclmdlSetAIPPPixelVarReci(aippDynamicSet,
            dyAippCfg->GetPixelVarReci()[pixelVarReciIndex].dtcPixelVarReciChn0,
            dyAippCfg->GetPixelVarReci()[pixelVarReciIndex].dtcPixelVarReciChn1,
            dyAippCfg->GetPixelVarReci()[pixelVarReciIndex].dtcPixelVarReciChn2,
            dyAippCfg->GetPixelVarReci()[pixelVarReciIndex].dtcPixelVarReciChn3, batchIndex);
    } else {
        DEBUG_LOG("aclmdlSetAIPPPixelVarReci params: aippDynamicSet: %p dtcPixelVarReciChn0: %f dtcPixelVarReciChn1: \
            %f dtcPixelVarReciChn2: %f dtcPixelVarReciChn3: %f batchIndex: %d", aippDynamicSet, 0.0,
            0.0, 0.0, 0.0, int(batchIndex));
        ret = aclmdlSetAIPPPixelVarReci(aippDynamicSet, 0.0, 0.0, 0.0, 0.0, batchIndex);
    }

    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPPixelVarReci failed, ret %d", ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPCropParams(
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    aclmdlAIPP* aippDynamicSet, size_t batchIndex
)
{
    aclError ret = ACL_ERROR_NONE;
    int cropIndex = GetDynamicAippParaByBatch(batchIndex, dyAippCfg, "crop");
    if (cropIndex >= 0) {
        DEBUG_LOG("aclmdlSetAIPPCropParams params: aippDynamicSet: %p cropSwitch: %d loadStartPosW: %d \
            loadStartPosH: %d cropSizeW: %d cropSizeH: %d batchIndex: %d", aippDynamicSet,
            dyAippCfg->GetCropParams()[cropIndex].cropSwitch, dyAippCfg->GetCropParams()[cropIndex].loadStartPosW,
            dyAippCfg->GetCropParams()[cropIndex].loadStartPosH, dyAippCfg->GetCropParams()[cropIndex].cropSizeW,
            dyAippCfg->GetCropParams()[cropIndex].cropSizeH, int(batchIndex));
        ret = aclmdlSetAIPPCropParams(aippDynamicSet, dyAippCfg->GetCropParams()[cropIndex].cropSwitch,
            dyAippCfg->GetCropParams()[cropIndex].loadStartPosW, dyAippCfg->GetCropParams()[cropIndex].loadStartPosH,
            dyAippCfg->GetCropParams()[cropIndex].cropSizeW, dyAippCfg->GetCropParams()[cropIndex].cropSizeH,
            batchIndex);
    } else {
        ret = aclmdlSetAIPPCropParams(aippDynamicSet, 0, 0, 0, Base::CROP_SIZE_W_DEFAULT,
            Base::CROP_SIZE_H_DEFAULT, batchIndex);
    }
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPCropParams failed, ret %d", ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetAIPPPaddingParams(
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    aclmdlAIPP* aippDynamicSet, size_t batchIndex
)
{
    aclError ret = ACL_ERROR_NONE;
    int padIndex = GetDynamicAippParaByBatch(batchIndex, dyAippCfg, "pad");
    if (padIndex >= 0) {
        DEBUG_LOG("aclmdlSetAIPPPaddingParams params: aippDynamicSet: %p paddingSwitch: %d paddingSizeTop: %d \
            paddingSizeBottom: %d paddingSizeLeft: %d paddingSizeRight: %d batchIndex: %d", aippDynamicSet,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSwitch,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSizeTop,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSizeBottom,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSizeLeft,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSizeRight,
            int(batchIndex));
        ret = aclmdlSetAIPPPaddingParams(aippDynamicSet, dyAippCfg->GetPaddingParams()[padIndex].paddingSwitch,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSizeTop,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSizeBottom,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSizeLeft,
            dyAippCfg->GetPaddingParams()[padIndex].paddingSizeRight,
            batchIndex);
    } else {
        ret = aclmdlSetAIPPPaddingParams(aippDynamicSet, 0, 0, 0, 0, 0, batchIndex);
    }
    if (ret != ACL_ERROR_NONE) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlSetAIPPPaddingParams failed, ret %d", ret);
        throw "AippData set failed!";
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::GetDymAIPPConfigSet(
    std::shared_ptr<Base::DynamicAippConfig> dyAippCfg,
    aclmdlAIPP* &pAIPPSet, uint64_t maxBatchSize
)
{
    Result ret = SUCCESS;
    INFO_LOG("dynamic aipp mode. batchsize:%d", int(maxBatchSize));

    aclmdlAIPP *aippDynamicSet = aclmdlCreateAIPP(maxBatchSize);
    if (aippDynamicSet == nullptr) {
        cout << aclGetRecentErrMsg() << endl;
        ERROR_LOG("aclmdlCreateAIPP failed");
        return FAILED;
    }
    try {
        ret = SetAIPPSrcImageSize(dyAippCfg, aippDynamicSet);
        ret = SetAIPPInputFormat(dyAippCfg, aippDynamicSet);
        ret = SetAIPPCscParams(dyAippCfg, aippDynamicSet);
        ret = SetAIPPRbuvSwapSwitch(dyAippCfg, aippDynamicSet);
        ret = SetAIPPAxSwapSwitch(dyAippCfg, aippDynamicSet);
    } catch (...) {
        FreeAIPP(aippDynamicSet);
        return FAILED;
    }

    for (size_t batchIndex = 0; batchIndex < maxBatchSize; batchIndex++) { // 遍历设置需要以batchIndex为单位的配置
        try {
            SetAIPPDtcPixelMean(dyAippCfg, aippDynamicSet, batchIndex);
            SetAIPPDtcPixelMin(dyAippCfg, aippDynamicSet, batchIndex);
            SetAIPPPixelVarReci(dyAippCfg, aippDynamicSet, batchIndex);
            SetAIPPCropParams(dyAippCfg, aippDynamicSet, batchIndex);
            SetAIPPPaddingParams(dyAippCfg, aippDynamicSet, batchIndex);
        } catch (...) {
            FreeAIPP(aippDynamicSet);
            return FAILED;
        }
    }
    if (pAIPPSet != nullptr) {
        FreeAIPP(pAIPPSet);
    }
    pAIPPSet = aippDynamicSet;
    DEBUG_LOG("debug now get pset :%p %p\n", pAIPPSet, aippDynamicSet);
    return ret;
}