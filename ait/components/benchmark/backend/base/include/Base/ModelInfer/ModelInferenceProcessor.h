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

#ifndef MODLE_INFERENCEPROCESSOR_H
#define MODLE_INFERENCEPROCESSOR_H

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "acl/acl.h"
#include "Base/ErrorCode/ErrorCode.h"
#include "Base/Tensor/TensorBase/TensorBase.h"

#include "Base/ModelInfer/SessionOptions.h"
#include "Base/ModelInfer/model_process.h"
#include "Base/ModelInfer/DynamicAippConfig.h"

#define CHECK_RET_EQ(func, expect_value) \
{ \
auto ret = (func); \
if (ret != (expect_value)) { \
    WARN_LOG("Check failed:%s, ret:%d\n", #func, ret); \
    return ret; \
} \
}

namespace Base {
struct BaseTensor {
    void* buf;
    std::vector<int64_t> shape;
    size_t size;
    size_t len;

    BaseTensor() = default;

    BaseTensor(int64_t buf, int64_t size)
    {
        this->buf = (void*)buf;
        this->size = (size_t)size;
    }

    BaseTensor(void* buf, size_t size)
    {
        this->buf = buf;
        this->size = size;
    }
};

struct TensorDesc
{
    std::string name;
    TensorDataType datatype;
    size_t format;
    std::vector<int64_t> shape;
    size_t size;
    size_t realsize;    // 针对动态shape 动态分档场景 实际需要的大小
};

enum DataFormat {
    NCHW = 0,
    NHWC = 1
};

enum DynamicType {
    STATIC_BATCH = 0,
    DYNAMIC_BATCH = 1,
    DYNAMIC_HW = 2,
    DYNAMIC_DIMS = 3,
    DYNAMIC_SHAPE = 4,
};

struct ImageSize {
    size_t height;
    size_t width;

    ImageSize() = default;

    ImageSize(size_t height, size_t width)
    {
        this->width = width;
        this->height = height;
    }
};

struct DyDimsInfo {
    std::vector<std::string> dym_dims;
};

struct DyShapeInfo {
    std::vector<int64_t> dims_num;
    std::map<string, std::vector<int64_t>> dym_shape_map;
};

struct DynamicInfo {
    DynamicType dynamicType = STATIC_BATCH;
    union {
        struct {
            uint64_t batchSize;
        }staticBatch;
        struct {
            uint64_t batchSize;
            uint64_t maxbatchsize;
        }dyBatch;
        struct {
            ImageSize imageSize;
            uint64_t maxHWSize;
        }dyHW;
        struct {
            DyDimsInfo* pDims;
        }dyDims;
        struct {
            DyShapeInfo* pShapes;
        }dyShape;
    };
};

struct ModelDesc {
    std::vector<Base::TensorDesc> inTensorsDesc;   // 所有 intensors信息 不包括dynamic index
    std::vector<Base::TensorDesc> outTensorsDesc;  // 所有out tensors信息

    std::map<std::string, size_t> innames2Index;

    std::map<std::string, size_t> outnames2Index;
};

struct InferSumaryInfo {
    std::vector<float> execTimeList;
};

class ModelInferenceProcessor {
public:
    /**
     * @description Init
     * 1.Loading  Model
     * 2.Get input sizes and output sizes
     * @return APP_ERROR error code
     */
    APP_ERROR Init(const std::string& modelPath, std::shared_ptr<SessionOptions> options, const int32_t &deviceId);

    /**
     * @description Unload Model
     * @return APP_ERROR error code
     */
    APP_ERROR DeInit(void);

    /**
     * @description ModelInference
     * 1.Get model description
     * 2.Execute model infer
     * @return APP_ERROR error code
     */
    APP_ERROR Inference(const std::vector<TensorBase>& feeds, std::vector<std::string> outputNames, std::vector<TensorBase>& outputTensors);

    APP_ERROR Inference(const std::map<std::string, TensorBase>& feeds, std::vector<std::string> outputNames, std::vector<TensorBase>& outputTensors);

    APP_ERROR Inference(const std::vector<BaseTensor>& feeds, std::vector<std::string> &outputNames, std::vector<TensorBase>& outputTensors);

    APP_ERROR RepeatInference(const std::vector<int>& inOutRelation)

    APP_ERROR ModelInference_Inner(std::vector<BaseTensor> &inputs, std::vector<std::string> outputNames, std::vector<TensorBase>& outputTensors);

    APP_ERROR RepeatInference(const std::vector<int>& inOutRelation)

    /**
     * @description get modelDesc
     */
    const std::vector<Base::TensorDesc>& GetInputs() const;
    const std::vector<Base::TensorDesc>& GetOutputs() const;

    std::shared_ptr<SessionOptions> GetOptions();

    APP_ERROR ResetSumaryInfo();
    const InferSumaryInfo& GetSumaryInfo();

    APP_ERROR SetStaticBatch();
    APP_ERROR SetDynamicBatchsize(int batchsize);
    APP_ERROR SetDynamicHW(int width, int height);
    APP_ERROR SetDynamicDims(std::string dymdimsStr);
    APP_ERROR SetDynamicShape(std::string dymshapeStr);
    APP_ERROR SetCustomOutTensorsSize(std::vector<size_t> customOutSize);

    uint64_t GetMaxDymBatchsize();
    int GetDymAIPPInputExist();
    APP_ERROR CheckDymAIPPInputExist();
    APP_ERROR SetDymAIPPInfoSet();

    APP_ERROR AippSetMaxBatchSize(uint64_t batchSize);
    APP_ERROR SetInputFormat(std::string iptFmt);
    APP_ERROR SetSrcImageSize(std::vector<int> srcImageSize);
    APP_ERROR SetRbuvSwapSwitch(int rsSwitch);
    APP_ERROR SetAxSwapSwitch(int asSwitch);
    APP_ERROR SetCscParams(std::vector<int> cscParams);
    APP_ERROR SetCropParams(std::vector<int> cropParams);
    APP_ERROR SetPaddingParams(std::vector<int> padParams);
    APP_ERROR SetDtcPixelMean(std::vector<int> meanParams);
    APP_ERROR SetDtcPixelMin(std::vector<float> minParams);
    APP_ERROR SetPixelVarReci(std::vector<float> reciParams);

private:

    APP_ERROR SetDynamicInfo();

    APP_ERROR AllocDyIndexMem();
    APP_ERROR AllocDymAIPPIndexMem();
    APP_ERROR FreeDyIndexMem();
    APP_ERROR FreeDymInfoMem();
    APP_ERROR FreeDymAIPPMem();

    APP_ERROR DestroyOutMemoryData(std::vector<MemoryData>& outputs);
    APP_ERROR CreateOutMemoryData(std::vector<MemoryData>& outputs);
    APP_ERROR AddOutTensors(std::vector<MemoryData>& outputs, std::vector<std::string> outputNames, std::vector<TensorBase>& outputTensors);

    APP_ERROR GetModelDescInfo();
    APP_ERROR DestroyInferCacheData();

    APP_ERROR SetInputsData(std::vector<BaseTensor> &inputs);
    APP_ERROR UpdateInputsData(std::vector<int> &inOutRelation);
    APP_ERROR SetAippConfigData();
    APP_ERROR Execute();
    APP_ERROR GetOutputs(std::vector<std::string> outputNames, std::vector<TensorBase> &outputTensors);

    APP_ERROR CheckInVectorAndFillBaseTensor(const std::vector<BaseTensor>& feeds, std::vector<BaseTensor> &inputs);
    APP_ERROR CheckInVectorAndFillBaseTensor(const std::vector<TensorBase>& feeds, std::vector<BaseTensor> &inputs);
    APP_ERROR CheckInMapAndFillBaseTensor(const std::map<std::string, TensorBase>& feeds, std::vector<BaseTensor> &inputs);

private:
    ModelDesc modelDesc_;

    InferSumaryInfo sumaryInfo_ = {};
    std::shared_ptr<ModelProcess> processModel;
    std::shared_ptr<DynamicAippConfig> dyAippCfg;
    DynamicInfo dynamicInfo_ = {};

    size_t dynamicIndex_ = -1;
    MemoryData dynamicIndexMemory_;

    std::map<size_t, MemoryData> dymAIPPIndexMemory_;
    std::map<size_t, aclmdlAIPP*> dymAIPPIndexSet_;

    size_t dym_gear_count_;

    std::shared_ptr<SessionOptions> options_;
    int32_t deviceId_;

    std::vector<size_t> customOutTensorSize_;
    std::vector<MemoryData> outputsMemDataQue_;
};
}  // namespace Base
#endif
