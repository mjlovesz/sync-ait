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

#ifndef PY_MODEL_INFFER
#define PY_MODEL_INFFER

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <thread>

#ifdef COMPILE_PYTHON_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include "Base/ModelInfer/SessionOptions.h"

#include "Base/ModelInfer/ModelInferenceProcessor.h"
#include "Base/Tensor/TensorBase/TensorBase.h"
#include "Base/ModelInfer/DynamicAippConfig.h"

namespace Base {
class PyInferenceSession
{
public:
    PyInferenceSession(const std::string &modelPath, const uint32_t &deviceId, std::shared_ptr<SessionOptions> options);
    ~PyInferenceSession();

    std::vector<TensorBase> InferMap(std::vector<std::string>& output_names, std::map<std::string, TensorBase>& feeds);
    std::vector<TensorBase> InferVector(std::vector<std::string>& output_names, std::vector<TensorBase>& feeds);

    std::vector<TensorBase> InferBaseTensorVector(std::vector<std::string>& output_names, std::vector<Base::BaseTensor>& feeds);
    void OnlyInfer(std::vector<BaseTensor> &inputs, std::vector<std::string>& output_names, std::vector<TensorBase>& outputs);
    void InferPipeline(std::vector<std::vector<std::string>>& infilesList, const std::string& outputDir,
                       bool autoDymShape, bool autoDymDims, const std::string& outFmt, const bool pureInferMode);
    std::vector<std::vector<TensorBase>> InferPipelineBaseTensor(std::vector<std::string>& outputNames,
                                                                 std::vector<std::vector<Base::BaseTensor>>& inputsList,
                                                                 std::vector<std::vector<std::vector<size_t>>>& shapesList,
                                                                 bool autoDymShape, bool autoDymDims);

    std::vector<std::vector<uint64_t>> GetDynamicHW();
    std::vector<int64_t> GetDynamicBatch();

    const std::vector<Base::TensorDesc>& GetInputs();
    const std::vector<Base::TensorDesc>& GetOutputs();

    uint32_t GetDeviceId() const;
    std::string GetDesc();

    std::shared_ptr<SessionOptions> GetOptions();

    const InferSumaryInfo& GetSumaryInfo();

    int ResetSumaryInfo();
    int SetStaticBatch();
    int SetDynamicBatchsize(int batchsize);
    int SetDynamicHW(int width, int height);
    int SetDynamicDims(std::string dymdimsStr);

    int SetDynamicShape(std::string dymshapeStr);
    int SetCustomOutTensorsSize(std::vector<size_t> customOutSize);

    uint64_t GetMaxDymBatchsize();
    int SetDymAIPPInfoSet();
    int GetDymAIPPInputExist();
    int CheckDymAIPPInputExist();

    int AippSetMaxBatchSize(uint64_t batchSize);
    int SetInputFormat(std::string iptFmt);
    int SetSrcImageSize(std::vector<int> srcImageSize);
    int SetRbuvSwapSwitch(int rsSwitch);
    int SetAxSwapSwitch(int asSwitch);
    int SetCscParams(std::vector<int> cscParams);
    int SetCropParams(std::vector<int> cropParams);
    int SetPaddingParams(std::vector<int> padParams);
    int SetDtcPixelMean(std::vector<int> meanParams);
    int SetDtcPixelMin(std::vector<float> minParams);
    int SetPixelVarReci(std::vector<float> reciParams);

    TensorBase CreateTensorFromFilesList(Base::TensorDesc &dstTensorDesc, std::vector<std::string>& filesList);

    int Finalize();

    Base::ModelInferenceProcessor modelInfer_ = {};

private:
    void Init(const std::string &modelPath, std::shared_ptr<SessionOptions> options);
    int Destroy();

private:
    uint32_t deviceId_ = 0;
    Base::ModelDesc modelDesc_ = {};
    bool InitFlag_ = false;
};
}

#ifdef COMPILE_PYTHON_MODULE
    void RegistInferenceSession(py::module &m);
    void RegistAippConfig(py::class_<Base::PyInferenceSession, std::shared_ptr<Base::PyInferenceSession>>& model);
#endif

#endif
