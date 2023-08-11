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

#include "PyInferenceSession/PyInferenceSession.h"

#include <exception>
#include <thread>

#include "Base/DeviceManager/DeviceManager.h"
#include "Base/Tensor/TensorBuffer/TensorBuffer.h"
#include "Base/Tensor/TensorShape/TensorShape.h"
#include "Base/Tensor/TensorContext/TensorContext.h"
#include "Base/ErrorCode/ErrorCode.h"
#include "Base/Log/Log.h"
#include "Base/ModelInfer/pipeline.h"

namespace Base {
PyInferenceSession::PyInferenceSession(const std::string &modelPath, const uint32_t &deviceId, std::shared_ptr<SessionOptions> options) : deviceId_(deviceId), modelPath_(modelPath)
{
    Init(modelPath, options);
}

int PyInferenceSession::Destroy()
{
    if (InitFlag_ == false) {
        return APP_ERR_OK;
    }
    APP_ERROR ret = TensorContext::GetInstance()->SetContext(deviceId_);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("TensorContext::SetContext failed. ret=%d", ret);
        return ret;
    }
    ret = modelInfer_.DeInit();
    if (ret != APP_ERR_OK) {
        ERROR_LOG("ModelInfer Deinit failed. ret=%d", ret);
        return ret;
    }
    DEBUG_LOG("PyInferSession DestroySession successfully!");
    InitFlag_ = false;
    return APP_ERR_OK;
}

int PyInferenceSession::Finalize()
{
    APP_ERROR ret = TensorContext::GetInstance()->SetContext(deviceId_);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("TensorContext::SetContext failed. ret=%d", ret);
        return ret;
    }

    ret = Destroy();
    if (ret != APP_ERR_OK) {
        ERROR_LOG("TensorContext::Finalize. ret=%d", ret);
        return ret;
    }
    ret = TensorContext::GetInstance()->Finalize();
    if (ret != APP_ERR_OK) {
        ERROR_LOG("TensorContext::Finalize. ret=%d", ret);
        return ret;
    }
    DEBUG_LOG("PyInferSession Finalize successfully!");
    return APP_ERR_OK;
}

PyInferenceSession::~PyInferenceSession()
{
    Destroy();
}

void PyInferenceSession::Init(const std::string &modelPath, std::shared_ptr<SessionOptions> options)
{
    DeviceManager::GetInstance()->SetAclJsonPath(options->aclJsonPath);

    APP_ERROR ret = TensorContext::GetInstance()->SetContext(deviceId_);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    ret = modelInfer_.Init(modelPath, options, deviceId_);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    InitFlag_ = true;
}

std::vector<TensorBase> PyInferenceSession::InferMap(std::vector<std::string>& output_names, std::map<std::string, TensorBase>& feeds)
{
    DEBUG_LOG("start to ModelInference feeds");

    std::vector<TensorBase> outputs = {};
    APP_ERROR ret = modelInfer_.Inference(feeds, output_names, outputs);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    return outputs;
}

std::vector<TensorBase> PyInferenceSession::InferVector(std::vector<std::string>& output_names, std::vector<TensorBase>& feeds)
{
    DEBUG_LOG("start to ModelInference");

    std::vector<TensorBase> outputs = {};
    APP_ERROR ret = modelInfer_.Inference(feeds, output_names, outputs);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    return outputs;
}

std::string GetShapeDesc(std::vector<int64_t> shape)
{
    std::string shapeStr = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shapeStr += std::to_string(shape.at(i));
        if (i != shape.size() - 1) {
            shapeStr += ", ";
        }
    }
    shapeStr += ")";
    return shapeStr;
}

std::string GetTensorDesc(Base::TensorDesc desc)
{
    return GetShapeDesc(desc.shape) + "  " + Base::GetTensorDataTypeDesc(desc.datatype) + "  " + std::to_string(desc.size) + "  " + std::to_string(desc.realsize);
}

uint32_t PyInferenceSession::GetDeviceId() const
{
    return deviceId_;
}

const std::vector<Base::TensorDesc>& PyInferenceSession::GetInputs()
{
    return modelInfer_.GetInputs();
}

const std::vector<Base::TensorDesc>& PyInferenceSession::GetOutputs()
{
    return modelInfer_.GetOutputs();
}

std::shared_ptr<SessionOptions> PyInferenceSession::GetOptions()
{
    return modelInfer_.GetOptions();
}

std::string PyInferenceSession::GetDesc()
{
    std::string inputStr = "input:\n";
    std::string outputStr = "output:\n";
    auto &inTensorsDesc = modelInfer_.GetInputs();

    for (size_t i = 0; i < inTensorsDesc.size(); ++i) {
        inputStr += "  #" + std::to_string(i) + "  ";
        inputStr += "  " + inTensorsDesc[i].name + "  ";
        inputStr += GetTensorDesc(inTensorsDesc[i]) + "\n";
    }

    auto &outTensorsDesc = modelInfer_.GetOutputs();
    for (size_t i = 0; i < outTensorsDesc.size(); ++i) {
        outputStr += "  #" + std::to_string(i) + "  ";
        outputStr += "  " + outTensorsDesc[i].name + "  ";
        outputStr += GetTensorDesc(outTensorsDesc[i]) + "\n";
    }

    return "<Model>\ndevice:\t" + std::to_string(GetDeviceId()) + "\n" + inputStr + outputStr;
}

const InferSumaryInfo& PyInferenceSession::GetSumaryInfo()
{
    return modelInfer_.GetSumaryInfo();
}

int PyInferenceSession::ResetSumaryInfo()
{
    APP_ERROR ret = modelInfer_.ResetSumaryInfo();
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetStaticBatch()
{
    APP_ERROR ret = modelInfer_.SetStaticBatch();
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDynamicBatchsize(int batchsize)
{
    APP_ERROR ret = modelInfer_.SetDynamicBatchsize(batchsize);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

uint64_t PyInferenceSession::GetMaxDymBatchsize()
{
    return modelInfer_.GetMaxDymBatchsize();
}

int PyInferenceSession::GetDymAIPPInputExist()
{
    return modelInfer_.GetDymAIPPInputExist();
}

int PyInferenceSession::CheckDymAIPPInputExist()
{
    APP_ERROR ret = modelInfer_.CheckDymAIPPInputExist();
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDymAIPPInfoSet()
{
    APP_ERROR ret = modelInfer_.SetDymAIPPInfoSet();
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDynamicHW(int width, int height)
{
    APP_ERROR ret = modelInfer_.SetDynamicHW(width, height);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDynamicDims(std::string dymdimsStr)
{
    APP_ERROR ret = modelInfer_.SetDynamicDims(dymdimsStr);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDynamicShape(std::string dymshapeStr)
{
    APP_ERROR ret = modelInfer_.SetDynamicShape(dymshapeStr);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetCustomOutTensorsSize(std::vector<size_t> customOutSize)
{
    APP_ERROR ret = modelInfer_.SetCustomOutTensorsSize(customOutSize);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

std::vector<TensorBase> PyInferenceSession::InferBaseTensorVector(std::vector<std::string>& output_names, std::vector<Base::BaseTensor>& feeds)
{
    DEBUG_LOG("start to ModelInference base_tensor");

    std::vector<MemoryData> memorys = {};
    std::vector<BaseTensor> inputs = {};
    for (auto &info : feeds) {
        MemoryData mem = CopyMemory2DeviceMemory(info.buf, info.size, deviceId_);
        memorys.push_back(mem);
        BaseTensor tensor(mem.ptrData, mem.size);
        inputs.push_back(tensor);
    }

    std::vector<TensorBase> outputs = {};
    APP_ERROR ret = modelInfer_.Inference(inputs, output_names, outputs);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    for (auto &mem : memorys) {
        MemoryHelper::Free(mem);
    }
    return outputs;
}

void PyInferenceSession::OnlyInfer(std::vector<BaseTensor> &inputs, std::vector<std::string>& output_names, std::vector<TensorBase>& outputs)
{
    APP_ERROR ret = modelInfer_.Inference(inputs, output_names, outputs);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
}

std::vector<std::vector<TensorBase>> PyInferenceSession::InferPipelineBaseTensor(
    std::vector<std::string>& outputNames, std::vector<std::vector<Base::BaseTensor>>& inputsList,
    std::vector<std::vector<std::vector<size_t>>>& shapesList, bool autoDymShape, bool autoDymDims)
{
    DEBUG_LOG("start to ModelInference base_tensor in pipeline");
    std::vector<std::vector<TensorBase>> result{};

    uint32_t deviceId = GetDeviceId();
    ConcurrentQueue<std::shared_ptr<Feeds>> h2dQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> computeQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> d2hQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> saveQueue;

    std::thread h2dThread(FuncH2d, std::ref(h2dQueue), std::ref(computeQueue), deviceId);
    std::thread computeThread(FuncCompute, std::ref(computeQueue), std::ref(d2hQueue), deviceId, this);
    std::thread d2hThread(FuncD2h, std::ref(d2hQueue), std::ref(saveQueue), deviceId);
    std::thread saveThread(FuncSaveTensorBase, std::ref(saveQueue), deviceId, std::ref(result));
    FuncPrepareBaseTensor(h2dQueue, deviceId, this, inputsList, shapesList, autoDymShape, autoDymDims, outputNames);

    h2dThread.join();
    computeThread.join();
    d2hThread.join();
    saveThread.join();

    return result;
}

void PyInferenceSession::InferPipeline(std::vector<std::vector<std::string>>& infilesList, const std::string& outputDir,
                                       bool autoDymShape, bool autoDymDims, const std::string& outFmt,
                                       const bool pureInferMode, size_t num_threads = 1)
{
    uint32_t deviceId = GetDeviceId();
    ConcurrentQueue<std::shared_ptr<Feeds>> h2dQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> computeQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> d2hQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> saveQueue;
    std::vector<std::thread> computeThreadGroup{};
    computeThreadGroup.reserve(num_threads-1);

    std::thread h2dThread(FuncH2d, std::ref(h2dQueue), std::ref(computeQueue), deviceId, num_threads);
    std::thread computeThread(FuncCompute, std::ref(computeQueue), std::ref(d2hQueue), deviceId, this);
    for (size_t i = 0; i < num_threads - 1; i++) {
        computeThreadGroup.emplace_back(FuncComputeWithoutSession, std::ref(computeQueue), std::ref(d2hQueue), deviceId, modelPath_)
    }
    std::thread d2hThread(FuncD2h, std::ref(d2hQueue), std::ref(saveQueue), deviceId, num_threads);
    std::thread saveThread(FuncSave, std::ref(saveQueue), deviceId, outFmt);
    FuncPrepare(h2dQueue, deviceId, this, infilesList, autoDymShape, autoDymDims, outputDir, pureInferMode);

    h2dThread.join();
    computeThread.join();
    for (auto &elem : computeThreadGroup) {
        elem.join();
    }
    d2hThread.join();
    saveThread.join();
}

int PyInferenceSession::AippSetMaxBatchSize(uint64_t batchSize)
{
    APP_ERROR ret = modelInfer_.AippSetMaxBatchSize(batchSize);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetInputFormat(std::string iptFmt)
{
    APP_ERROR ret = modelInfer_.SetInputFormat(iptFmt);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetSrcImageSize(std::vector<int> srcImageSize)
{
    APP_ERROR ret = modelInfer_.SetSrcImageSize(srcImageSize);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetRbuvSwapSwitch(int rsSwitch)
{
    APP_ERROR ret = modelInfer_.SetRbuvSwapSwitch(rsSwitch);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetAxSwapSwitch(int asSwitch)
{
    APP_ERROR ret = modelInfer_.SetAxSwapSwitch(asSwitch);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetCscParams(std::vector<int> cscParams)
{
    APP_ERROR ret = modelInfer_.SetCscParams(cscParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetCropParams(std::vector<int> cropParams)
{
    APP_ERROR ret = modelInfer_.SetCropParams(cropParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetPaddingParams(std::vector<int> padParams)
{
    APP_ERROR ret = modelInfer_.SetPaddingParams(padParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDtcPixelMean(std::vector<int> meanParams)
{
    APP_ERROR ret = modelInfer_.SetDtcPixelMean(meanParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDtcPixelMin(std::vector<float> minParams)
{
    APP_ERROR ret = modelInfer_.SetDtcPixelMin(minParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetPixelVarReci(std::vector<float> reciParams)
{
    APP_ERROR ret = modelInfer_.SetPixelVarReci(reciParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

TensorBase PyInferenceSession::CreateTensorFromFilesList(Base::TensorDesc &dstTensorDesc, std::vector<std::string>& filesList)
{
    std::vector<uint32_t> u32shape;
    for (size_t j = 0; j < dstTensorDesc.shape.size(); ++j) {
        u32shape.push_back((uint32_t)(dstTensorDesc.shape[j]));
    }
    // malloc
    TensorBase dstTensor = TensorBase(u32shape, dstTensorDesc.datatype,
        MemoryData::MemoryType::MEMORY_HOST, -1);
    APP_ERROR ret = Base::TensorBase::TensorBaseMalloc(dstTensor);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("TensorBaseMalloc failed ret:%d", ret);
        throw std::runtime_error(GetError(ret));
    }
    // copy
    size_t offset = 0;
    char *ptr = (char *)dstTensor.GetBuffer();
    for (uint32_t i = 0; i < filesList.size(); i++) {
        Result ret = Utils::FillFileContentToMemory(filesList[i], ptr, dstTensor.GetByteSize(), offset);
        if (ret != SUCCESS) {
            ERROR_LOG("TensorBaseMalloc i:%d file:%s failed ret:%d", i, filesList[i].c_str(), ret);
            throw std::runtime_error(GetError(ret));
        }
    }
    dstTensor.ToDevice(deviceId_);
    return dstTensor;
}
}

std::shared_ptr<Base::PyInferenceSession> CreateModelInstance(const std::string &modelPath, const uint32_t &deviceId, std::shared_ptr<Base::SessionOptions> options)
{
    return std::make_shared<Base::PyInferenceSession>(modelPath, deviceId, options);
}

#ifdef COMPILE_PYTHON_MODULE
void RegistInferenceSession(py::module &m)
{
    using namespace pybind11::literals;

    py::class_<Base::BaseTensor>(m, "BaseTensor")
        .def(py::init<int64_t, int64_t>())
        .def_readwrite("buf", &Base::BaseTensor::buf)
        .def_readwrite("size", &Base::BaseTensor::size);

    py::class_<Base::SessionOptions, std::shared_ptr<Base::SessionOptions>>(m, "session_options")
    .def(py::init([]() { return std::make_shared<Base::SessionOptions>(); }))
    .def_readwrite("loop", &Base::SessionOptions::loop)
    .def_readwrite("log_level", &Base::SessionOptions::log_level)
    .def_readwrite("acl_json_path", &Base::SessionOptions::aclJsonPath);

    py::class_<Base::TensorDesc>(m, "tensor_desc")
    .def(pybind11::init<>())
    .def_readwrite("name", &Base::TensorDesc::name)
    .def_readwrite("datatype", &Base::TensorDesc::datatype)
    .def_readwrite("format", &Base::TensorDesc::format)
    .def_readwrite("shape", &Base::TensorDesc::shape)
    .def_readwrite("realsize", &Base::TensorDesc::realsize)
    .def_readwrite("size", &Base::TensorDesc::size);

    py::class_<Base::InferSumaryInfo>(m, "sumary")
    .def(pybind11::init<>())
    .def_readwrite("exec_time_list", &Base::InferSumaryInfo::execTimeList);

    auto model = py::class_<Base::PyInferenceSession, std::shared_ptr<Base::PyInferenceSession>>(m, "InferenceSession");
    model.def(py::init<std::string&, int, std::shared_ptr<Base::SessionOptions>>());
    model.def("run", &Base::PyInferenceSession::InferVector);
    model.def("run", &Base::PyInferenceSession::InferMap);
    model.def("run", &Base::PyInferenceSession::InferBaseTensorVector);
    model.def("run_pipeline", &Base::PyInferenceSession::InferPipeline);
    model.def("run_pipeline", &Base::PyInferenceSession::InferPipelineBaseTensor);
    model.def("__str__", &Base::PyInferenceSession::GetDesc);
    model.def("__repr__", &Base::PyInferenceSession::GetDesc);

    model.def("options", &Base::PyInferenceSession::GetOptions, py::return_value_policy::reference);

    model.def("sumary", &Base::PyInferenceSession::GetSumaryInfo, py::return_value_policy::reference);
    model.def("get_inputs", &Base::PyInferenceSession::GetInputs, py::return_value_policy::reference);
    model.def("get_outputs", &Base::PyInferenceSession::GetOutputs, py::return_value_policy::reference);
    model.def("reset_sumaryinfo", &Base::PyInferenceSession::ResetSumaryInfo);
    model.def("set_staticbatch", &Base::PyInferenceSession::SetStaticBatch);
    model.def("set_dynamic_batchsize", &Base::PyInferenceSession::SetDynamicBatchsize);
    model.def("set_dynamic_hw", &Base::PyInferenceSession::SetDynamicHW);
    model.def("set_dynamic_dims", &Base::PyInferenceSession::SetDynamicDims);
    model.def("set_dynamic_shape", &Base::PyInferenceSession::SetDynamicShape);
    model.def("set_custom_outsize", &Base::PyInferenceSession::SetCustomOutTensorsSize);

    model.def("create_tensor_from_fileslist", &Base::PyInferenceSession::CreateTensorFromFilesList);
    model.def("finalize", &Base::PyInferenceSession::Finalize);
    RegistAippConfig(model);

    m.def("model", &CreateModelInstance, "modelPath"_a, "deviceId"_a = 0, "options"_a=py::none());
}

void RegistAippConfig(py::class_<Base::PyInferenceSession, std::shared_ptr<Base::PyInferenceSession>>& model)
{
    using namespace pybind11::literals;

    model.def("get_max_dym_batchsize", &Base::PyInferenceSession::GetMaxDymBatchsize);
    model.def("get_dym_aipp_input_exist", &Base::PyInferenceSession::GetDymAIPPInputExist);
    model.def("check_dym_aipp_input_exist", &Base::PyInferenceSession::CheckDymAIPPInputExist);
    model.def("set_dym_aipp_info_set", &Base::PyInferenceSession::SetDymAIPPInfoSet);

    model.def("aipp_set_max_batch_size", &Base::PyInferenceSession::AippSetMaxBatchSize);
    model.def("aipp_set_input_format", &Base::PyInferenceSession::SetInputFormat);
    model.def("aipp_set_src_image_size", &Base::PyInferenceSession::SetSrcImageSize);
    model.def("aipp_set_rbuv_swap_switch", &Base::PyInferenceSession::SetRbuvSwapSwitch);
    model.def("aipp_set_ax_swap_switch", &Base::PyInferenceSession::SetAxSwapSwitch);
    model.def("aipp_set_csc_params", &Base::PyInferenceSession::SetCscParams);
    model.def("aipp_set_crop_params", &Base::PyInferenceSession::SetCropParams);
    model.def("aipp_set_padding_params", &Base::PyInferenceSession::SetPaddingParams);
    model.def("aipp_set_dtc_pixel_mean", &Base::PyInferenceSession::SetDtcPixelMean);
    model.def("aipp_set_dtc_pixel_min", &Base::PyInferenceSession::SetDtcPixelMin);
    model.def("aipp_set_pixel_var_reci", &Base::PyInferenceSession::SetPixelVarReci);
}
#endif
