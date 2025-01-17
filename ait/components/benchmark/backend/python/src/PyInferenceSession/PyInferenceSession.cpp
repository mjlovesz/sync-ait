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
#include <set>

#include "Base/DeviceManager/DeviceManager.h"
#include "Base/Tensor/TensorBuffer/TensorBuffer.h"
#include "Base/Tensor/TensorShape/TensorShape.h"
#include "Base/Tensor/TensorContext/TensorContext.h"
#include "Base/ErrorCode/ErrorCode.h"
#include "Base/Log/Log.h"
#include "Base/ModelInfer/pipeline.h"

namespace Base {
PyInferenceSession::PyInferenceSession(const std::string &modelPath, const uint32_t &deviceId,
    std::shared_ptr<SessionOptions> options)
    : deviceId_(deviceId), modelPath_(modelPath)
{
    Init(modelPath, options);
}

void PyInferenceSession::SetContext()
{
    APP_ERROR ret = TensorContext::GetInstance()->SetContext(deviceId_, contextIndex_);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
}

int PyInferenceSession::Destroy()
{
    if (InitFlag_ == false) {
        return APP_ERR_OK;
    }
    APP_ERROR ret = TensorContext::GetInstance()->SetContext(deviceId_, contextIndex_);
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
    APP_ERROR ret = TensorContext::GetInstance()->Finalize();
    if (ret != APP_ERR_OK) {
        ERROR_LOG("TensorContext::Finalize failed. ret=%d", ret);
        return ret;
    }
    DEBUG_LOG("PyInferSession Finalize successfully!");
    return APP_ERR_OK;
}

int PyInferenceSession::FreeResource()
{
    APP_ERROR ret = TensorContext::GetInstance()->SetContext(deviceId_, contextIndex_);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("TensorContext::SetContext failed. ret=%d", ret);
        return ret;
    }

    ret = Destroy();
    if (ret != APP_ERR_OK) {
        ERROR_LOG("Destroy failed. ret=%d", ret);
        return ret;
    }
    ret = TensorContext::GetInstance()->DestroyContext(deviceId_, contextIndex_);
    if (ret != APP_ERR_OK) {
        ERROR_LOG("TensorContext::DestroyContext. ret=%d", ret);
        return ret;
    }
    DEBUG_LOG("PyInferSession FreeResource successfully!");
    return APP_ERR_OK;
}


PyInferenceSession::~PyInferenceSession()
{
    Destroy();
}

void PyInferenceSession::Init(const std::string &modelPath, std::shared_ptr<SessionOptions> options)
{
    SETLOGLEVEL(options->log_level);
    DeviceManager::GetInstance()->SetAclJsonPath(options->aclJsonPath);
    APP_ERROR ret = TensorContext::GetInstance()->CreateContext(deviceId_, contextIndex_);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    SetContext();

    ret = modelInfer_.Init(modelPath, options, deviceId_, contextIndex_);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    InitFlag_ = true;
}

std::vector<TensorBase> PyInferenceSession::InferMap(std::vector<std::string>& output_names,
    std::map<std::string, TensorBase>& feeds)
{
    SetContext();
    DEBUG_LOG("start to ModelInference feeds");

    std::vector<TensorBase> outputs = {};
    APP_ERROR ret = modelInfer_.Inference(feeds, output_names, outputs);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    return outputs;
}

std::vector<TensorBase> PyInferenceSession::InferVector(std::vector<std::string>& output_names,
    std::vector<TensorBase>& feeds)
{
    SetContext();
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
    return (GetShapeDesc(desc.shape) + "  " + Base::GetTensorDataTypeDesc(desc.datatype) +
        "  " + std::to_string(desc.size) + "  " + std::to_string(desc.realsize));
}

uint32_t PyInferenceSession::GetDeviceId() const
{
    return deviceId_;
}

std::size_t PyInferenceSession::GetContextIndex() const
{
    return contextIndex_;
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

std::string PyInferenceSession::GetModelPath()
{
    return modelPath_;
}

std::string PyInferenceSession::GetDesc()
{
    SetContext();
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

const InferSumaryInfo& PyInferenceSession::GetSumaryInfo() const
{
    return modelInfer_.GetSumaryInfo();
}

void PyInferenceSession::MergeSummaryInfo(const InferSumaryInfo& summaryInfo)
{
    InferSumaryInfo& lhsSummaryInfo = modelInfer_.GetMutableSumaryInfo();
    lhsSummaryInfo.execTimeList.reserve(lhsSummaryInfo.execTimeList.size() + summaryInfo.execTimeList.size());
    for (auto time : summaryInfo.execTimeList) {
        lhsSummaryInfo.execTimeList.push_back(time);
    }
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
    SetContext();
    APP_ERROR ret = modelInfer_.SetStaticBatch();
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDynamicBatchsize(int batchsize)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetDynamicBatchsize(batchsize);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

uint64_t PyInferenceSession::GetMaxDymBatchsize()
{
    SetContext();
    return modelInfer_.GetMaxDymBatchsize();
}

int PyInferenceSession::GetDymAIPPInputExist()
{
    SetContext();
    return modelInfer_.GetDymAIPPInputExist();
}

int PyInferenceSession::CheckDymAIPPInputExist()
{
    SetContext();
    APP_ERROR ret = modelInfer_.CheckDymAIPPInputExist();
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDymAIPPInfoSet()
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetDymAIPPInfoSet();
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDynamicHW(int width, int height)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetDynamicHW(width, height);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDynamicDims(std::string dymdimsStr)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetDynamicDims(dymdimsStr);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDynamicShape(std::string dymshapeStr)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetDynamicShape(dymshapeStr);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetCustomOutTensorsSize(std::vector<size_t> customOutSize)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetCustomOutTensorsSize(customOutSize);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

std::vector<TensorBase> PyInferenceSession::InferBaseTensorVector(std::vector<std::string>& output_names,
    std::vector<Base::BaseTensor>& feeds)
{
    SetContext();
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

std::vector<TensorBase> PyInferenceSession::FirstInnerInfer(std::vector<std::string>& output_names,
    std::vector<Base::BaseTensor>& feeds)
{
    SetContext();
    DEBUG_LOG("start to FirstInnerInfer base_tensor");

    std::vector<MemoryData> memorys = {};
    std::vector<BaseTensor> inputs = {};
    for (auto &info : feeds) {
        MemoryData mem = CopyMemory2DeviceMemory(info.buf, info.size, deviceId_);
        memorys.push_back(mem);
        BaseTensor tensor(mem.ptrData, mem.size);
        inputs.push_back(tensor);
    }

    std::vector<TensorBase> outputs = {};
    APP_ERROR ret = modelInfer_.FirstInference(inputs, output_names, outputs);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    for (auto &mem : memorys) {
        MemoryHelper::Free(mem);
    }
    return outputs;
}

std::vector<TensorBase> PyInferenceSession::InnerInfer(const std::vector<int>& in_out_list,
    std::vector<std::string>& output_names, const bool get_outputs, const bool mem_copy)
{
    SetContext();
    std::vector<TensorBase> outputs = {};
    APP_ERROR ret = modelInfer_.RepeatInference(in_out_list, output_names, outputs, get_outputs, mem_copy);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return outputs;
}

void PyInferenceSession::OnlyInfer(std::vector<BaseTensor> &inputs, std::vector<std::string>& output_names,
    std::vector<TensorBase>& outputs)
{
    SetContext();
    APP_ERROR ret = modelInfer_.Inference(inputs, output_names, outputs);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
}

bool CheckExtraSession(size_t contextIndex, const std::vector<std::shared_ptr<PyInferenceSession>>& extraSession)
{
    std::set<size_t> contextSet{contextIndex};
    for (auto session : extraSession) {
        auto newContextIndex = session->GetContextIndex();
        if (contextSet.find(newContextIndex) != contextSet.end()) {
            return false;
        }
        contextSet.insert(newContextIndex);
    }
    return true;
}

std::vector<std::vector<TensorBase>> PyInferenceSession::InferPipelineBaseTensor(
    std::vector<std::string>& outputNames, std::vector<std::vector<Base::BaseTensor>>& inputsList,
    std::vector<std::vector<std::vector<size_t>>>& shapesList, bool autoDymShape, bool autoDymDims)
{
    SetContext();
    DEBUG_LOG("start to ModelInference base_tensor in pipeline");
    std::vector<std::vector<TensorBase>> result{};

    uint32_t deviceId = GetDeviceId();
    ConcurrentQueue<std::shared_ptr<Feeds>> h2dQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> computeQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> d2hQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> saveQueue;

    std::thread h2dThread(FuncH2d, std::ref(h2dQueue), std::ref(computeQueue), this);
    std::thread computeThread(FuncCompute, std::ref(computeQueue), std::ref(d2hQueue), this, nullptr);
    std::thread d2hThread(FuncD2h, std::ref(d2hQueue), std::ref(saveQueue), this);
    std::thread saveThread(FuncSaveTensorBase, std::ref(saveQueue), std::ref(result), this);
    FuncPrepareBaseTensor(h2dQueue, deviceId, this, inputsList, shapesList, autoDymShape, autoDymDims, outputNames);

    h2dThread.join();
    computeThread.join();
    d2hThread.join();
    saveThread.join();

    return result;
}

void PyInferenceSession::InferPipeline(std::vector<std::vector<std::string>>& infilesList,
                                       std::shared_ptr<InferOptions> inferOption,
                                       std::vector<std::shared_ptr<PyInferenceSession>>& extraSession)
{
    SetContext();
    if (!CheckExtraSession(contextIndex_, extraSession)) {
        ERROR_LOG("InferPipeline failed: cannot have session in same context");
        return;
    }
    size_t numThreads = extraSession.size() + 1;
    std::vector<ConcurrentQueue<std::shared_ptr<Feeds>>> h2dQueues(numThreads);
    std::vector<ConcurrentQueue<std::shared_ptr<Feeds>>> computeQueues(numThreads);
    std::vector<ConcurrentQueue<std::shared_ptr<Feeds>>> d2hQueues(numThreads);
    std::vector<ConcurrentQueue<std::shared_ptr<Feeds>>> saveQueues(numThreads);
    std::vector<std::thread> prepareThreadGroup{};
    std::vector<std::thread> h2dThreadGroup{};
    std::vector<std::thread> computeThreadGroup{};
    std::vector<std::thread> d2hThreadGroup{};
    std::vector<std::thread> saveThreadGroup{};
    std::vector<InferSumaryInfo> summaryInfoGroup(numThreads - 1);

    for (size_t i = 0; i < numThreads; i++) {
        Base::PyInferenceSession* session = this;
        InferSumaryInfo* inferSummary = nullptr;
        if (i != 0) {
            session = extraSession[i-1].get();
            inferSummary = &(summaryInfoGroup[i-1]);
            session->modelInfer_.GetMutableSumaryInfo().zero_point = this->GetSumaryInfo().zero_point;
        }
        prepareThreadGroup.emplace_back(FuncPrepare, std::ref(h2dQueues[i]), session, std::ref(infilesList),
            inferOption, numThreads, i);
        h2dThreadGroup.emplace_back(FuncH2d, std::ref(h2dQueues[i]), std::ref(computeQueues[i]), session);
        computeThreadGroup.emplace_back(FuncCompute, std::ref(computeQueues[i]), std::ref(d2hQueues[i]),
            session, inferSummary);
        d2hThreadGroup.emplace_back(FuncD2h, std::ref(d2hQueues[i]), std::ref(saveQueues[i]), session);
        saveThreadGroup.emplace_back(FuncSave, std::ref(saveQueues[i]), inferOption);
    }

    for (size_t i = 0; i < numThreads; i++) {
        prepareThreadGroup[i].join();
        h2dThreadGroup[i].join();
        computeThreadGroup[i].join();
        d2hThreadGroup[i].join();
        saveThreadGroup[i].join();
    }
    for (auto &summaryInfo : summaryInfoGroup) {
        MergeSummaryInfo(summaryInfo);
    }
}

int PyInferenceSession::AippSetMaxBatchSize(uint64_t batchSize)
{
    SetContext();
    APP_ERROR ret = modelInfer_.AippSetMaxBatchSize(batchSize);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetInputFormat(std::string iptFmt)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetInputFormat(iptFmt);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetSrcImageSize(std::vector<int> srcImageSize)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetSrcImageSize(srcImageSize);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetRbuvSwapSwitch(int rsSwitch)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetRbuvSwapSwitch(rsSwitch);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetAxSwapSwitch(int asSwitch)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetAxSwapSwitch(asSwitch);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetCscParams(std::vector<int> cscParams)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetCscParams(cscParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetCropParams(std::vector<int> cropParams)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetCropParams(cropParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetPaddingParams(std::vector<int> padParams)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetPaddingParams(padParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDtcPixelMean(std::vector<int> meanParams)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetDtcPixelMean(meanParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetDtcPixelMin(std::vector<float> minParams)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetDtcPixelMin(minParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

int PyInferenceSession::SetPixelVarReci(std::vector<float> reciParams)
{
    SetContext();
    APP_ERROR ret = modelInfer_.SetPixelVarReci(reciParams);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    return APP_ERR_OK;
}

TensorBase PyInferenceSession::CreateTensorFromFilesList(Base::TensorDesc &dstTensorDesc,
    std::vector<std::string>& filesList)
{
    SetContext();
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

namespace {
std::shared_ptr<Base::PyInferenceSession> CreateModelInstance(const std::string &modelPath,
    const uint32_t &deviceId, std::shared_ptr<Base::SessionOptions> options)
{
    return std::make_shared<Base::PyInferenceSession>(modelPath, deviceId, options);
}
} // namespace

#ifdef COMPILE_PYTHON_MODULE
void RegistTensor(py::module &m)
{
    using namespace pybind11::literals;

    py::class_<Base::BaseTensor>(m, "BaseTensor")
    .def(py::init<int64_t, int64_t>())
    .def_readwrite("buf", &Base::BaseTensor::buf)
    .def_readwrite("size", &Base::BaseTensor::size);

    py::class_<Base::TensorDesc>(m, "tensor_desc")
    .def(pybind11::init<>())
    .def_readwrite("name", &Base::TensorDesc::name)
    .def_readwrite("datatype", &Base::TensorDesc::datatype)
    .def_readwrite("format", &Base::TensorDesc::format)
    .def_readwrite("shape", &Base::TensorDesc::shape)
    .def_readwrite("realsize", &Base::TensorDesc::realsize)
    .def_readwrite("size", &Base::TensorDesc::size);
}
void RegistOptions(py::module &m)
{
    using namespace pybind11::literals;

    py::class_<Base::SessionOptions, std::shared_ptr<Base::SessionOptions>>(m, "session_options")
    .def(py::init([]() { return std::make_shared<Base::SessionOptions>(); }))
    .def_readwrite("loop", &Base::SessionOptions::loop)
    .def_readwrite("log_level", &Base::SessionOptions::log_level)
    .def_readwrite("acl_json_path", &Base::SessionOptions::aclJsonPath);

    py::class_<Base::InferOptions, std::shared_ptr<Base::InferOptions>>(m, "infer_options")
    .def(py::init([]() { return std::make_shared<Base::InferOptions>(); }))
    .def_readwrite("output_dir", &Base::InferOptions::outputDir)
    .def_readwrite("auto_dym_shape", &Base::InferOptions::autoDymShape)
    .def_readwrite("auto_dym_dims", &Base::InferOptions::autoDymDims)
    .def_readwrite("out_format", &Base::InferOptions::outFmt)
    .def_readwrite("pure_infer_mode", &Base::InferOptions::pureInferMode)
    .def_readwrite("output_names", &Base::InferOptions::outputNames)
    .def_readwrite("shapes_list", &Base::InferOptions::shapesList);
}

void RegistInferenceSession(py::module &m)
{
    using namespace pybind11::literals;

    py::class_<Base::InferSumaryInfo>(m, "sumary")
    .def(pybind11::init<>())
    .def_readwrite("exec_time_list", &Base::InferSumaryInfo::execTimeList);

    auto model = py::class_<Base::PyInferenceSession, std::shared_ptr<Base::PyInferenceSession>>(m, "InferenceSession");
    model.def(py::init<std::string&, int, std::shared_ptr<Base::SessionOptions>>());
    model.def("run", &Base::PyInferenceSession::InferVector);
    model.def("run", &Base::PyInferenceSession::InferMap);
    model.def("run", &Base::PyInferenceSession::InferBaseTensorVector);
    model.def("first_inner_run", &Base::PyInferenceSession::FirstInnerInfer);
    model.def("inner_run", &Base::PyInferenceSession::InnerInfer);
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
    model.def("free_resource", &Base::PyInferenceSession::FreeResource);
    model.def_static("finalize", &Base::PyInferenceSession::Finalize);

    RegistAippConfig(model);
    RegistOptions(m);
    RegistTensor(m);

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
