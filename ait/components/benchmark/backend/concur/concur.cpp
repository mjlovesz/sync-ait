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

#include <iostream>
#include <random>
#include <memory>
#include <mutex>
#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <unordered_map>

#include <unistd.h>
#include <assert.h>

#include "Base/Tensor/TensorBuffer/TensorBuffer.h"
#include "Base/Tensor/TensorShape/TensorShape.h"
#include "Base/Tensor/TensorContext/TensorContext.h"
#include "Base/Tensor/TensorBase/TensorBase.h"
#include "Base/ModelInfer/SessionOptions.h"
#include "Base/ModelInfer/ModelInferenceProcessor.h"
#include "Base/Log/Log.h"
#include "PyInferenceSession/PyInferenceSession.h"
#include "PyTensor/PyTensor.h"

#include "cnpy.h"
#include "utils.h"

namespace chr = std::chrono;
using TimePointPair = std::pair<chr::steady_clock::time_point, chr::steady_clock::time_point>;
using Arguments = std::unordered_map<std::string, std::string>;

struct Feeds {
    std::shared_ptr<std::vector<std::string>> outputNames = nullptr;
    std::shared_ptr<std::vector<Base::BaseTensor>> inputs = nullptr;
    std::shared_ptr<std::vector<Base::TensorBase>> outputs = nullptr;
    std::shared_ptr<std::vector<Base::MemoryData>> memory = nullptr;
    std::shared_ptr<std::vector<std::shared_ptr<cnpy::NpyArray>>> arrayPtr = nullptr;
    std::string autoDynamicShape = "";
    std::string outputPrefix = "";
};

template <typename T>
class ConcurrentQueue {
public:
    explicit ConcurrentQueue(int depth = 3): depth(depth) {}

    T pop() {
        std::unique_lock<std::mutex> mlock(mtx);
        while (queue.empty()) {
            condVar.wait(mlock);
        }
        auto val = queue.front();
        queue.pop();
        mlock.unlock();
        condVar.notify_one();
        return val;
    }

    void push (const T &item) {
        std::unique_lock<std::mutex> mlock(mtx);
        while (queue.size() >= depth) {
            condVar.wait(mlock);
        }
        queue.push(item);
        mlock.unlock();
        condVar.notify_one();
    }

private:
    std::mutex mtx;
    std::queue<T> queue;
    std::condition_variable condVar;
    int depth;
};

void SetSession(std::shared_ptr<Base::PyInferenceSession> session, Arguments& arguments)
{
    if (arguments["dymHW"] != "") {
        auto dymhw = SplitStr(arguments["dymHW"], ',');
        session->SetDynamicHW(stoi(dymhw[0]), stoi(dymhw[1]));
    }
    if (arguments["dymDims"] != "") {
        session->SetDynamicDims(arguments["dymDims"]);
    }
    if (arguments["dymShape"] != "") {
        session ->SetDynamicShape(arguments["dymShape"]);
    }
    if (arguments["outputSize"] != "") {
        auto outputSize = StrVecToNumVec(SplitStr(arguments["outputSize"], ','));
        session->SetCustomOutTensorsSize(outputSize);
    }
}

void FuncPrepare(int32_t deviceId, std::shared_ptr<Base::PyInferenceSession> session, std::string modelPath,
                 std::shared_ptr<Base::SessionOptions> options, std::vector<std::vector<std::string>> &filesList,
                 ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue, bool autoDymShape)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    std::vector<std::string> inputNames {};
    if (autoDymShape) {
        auto intensor_desc = session->GetInputs();
        for (auto &desc : intensor_desc) {
            inputNames.push_back(desc.name);
        }
    }

    for (auto &files : filesList) {
        auto outputNames = std::make_shared<std::vector<std::string>>();
        for (const auto &desc: session->GetOutputs()) {
            outputNames->push_back(desc.name);
        }
        auto inputs = std::make_shared<std::vector<Base::BaseTensor>>();
        auto arrayPtr = std::make_shared<std::vector<std::shared_ptr<cnpy::NpyArray>>>();
        std::string autoDynamicShape {};
        for (size_t i = 0; i < files.size(); i++) {
            auto array = std::make_shared<cnpy::NpyArray>(cnpy::NpyLoad(files[i]));
            arrayPtr->emplace_back(array);
            inputs->emplace_back(array->Data<void>(), array->NumBytes());
            if (autoDymShape) {
                autoDynamicShape += CreateDynamicShape(inputNames[i], array->shape);
                if (i != files.size()-1) {
                    autoDynamicShape += ";";
                }
            }
        }
        auto outputPrefix = GetPrefix(files.front());
        auto feeds = std::make_shared<Feeds>();
        feeds->autoDynamicShape = autoDynamicShape;
        feeds->arrayPtr = arrayPtr;
        feeds->outputNames = outputNames;
        feeds->inputs = inputs;
        feeds->outputPrefix = outputPrefix;
        h2dQueue.push(feeds);
    }
    h2dQueue.push(nullptr);
}

void FuncH2d(ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue,
             ConcurrentQueue<std::shared_ptr<Feeds>> &computeQueue, int32_t deviceId,
             std::shared_ptr<Base::PyInferenceSession> session,
             std::vector<TimePointPair>& timeStamps)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    while (true)
    {
        auto item = h2dQueue.pop();
        if (!item) {
            computeQueue.push(nullptr);
            break;
        }
        auto start = chr::steady_clock::now();

        item->memory = std::make_shared<std::vector<Base::MemoryData>>();
        auto inputs = std::make_shared<std::vector<Base::BaseTensor>>();
        for (auto &info : *(item->inputs)) {
            Base::MemoryData mem = Base::CopyMemory2DeviceMemory(info.buf, info.size, deviceId);
            item->memory->push_back(mem);
            Base::BaseTensor tensor(mem.ptrData, mem.size);
            inputs->push_back(tensor);
        }
        item->inputs = inputs;
        auto end = chr::steady_clock::now();
        timeStamps.emplace_back(start, end);

        computeQueue.push(item);
    }
}

void FuncCompute(ConcurrentQueue<std::shared_ptr<Feeds>> &computeQueue,
                 ConcurrentQueue<std::shared_ptr<Feeds>> &d2hQueue, int32_t deviceId,
                 std::shared_ptr<Base::PyInferenceSession> session,
                 std::vector<TimePointPair>& timeStamps)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    while (true)
    {
        auto item = computeQueue.pop();
        if (!item) {
            d2hQueue.push(nullptr);
            break;
        }
        auto start = chr::steady_clock::now();

        if (item->autoDynamicShape != "") {
            session->SetDynamicShape(item->autoDynamicShape);
        }
        auto outputs = std::make_shared<std::vector<Base::TensorBase>>();
        session->PureInfer(*(item->inputs), *(item->outputNames), *outputs);
        item->outputs = outputs;

        auto end = chr::steady_clock::now();
        timeStamps.emplace_back(start, end);

        d2hQueue.push(item);
    }
}

void FuncD2h(ConcurrentQueue<std::shared_ptr<Feeds>> &d2hQueue,
             ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue, int32_t deviceId,
             std::vector<TimePointPair>& timeStamps)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    while (true)
    {
        auto item = d2hQueue.pop();
        if (!item) {
            saveQueue.push(nullptr);
            break;
        }
        auto start = chr::steady_clock::now();

        for (auto &output : *(item->outputs)) {
            Base::TensorToHost(output);
        }

        auto end = chr::steady_clock::now();
        timeStamps.emplace_back(start, end);

        saveQueue.push(item);
    }
}

int TensotToNumpy(std::string outputFileName, Base::TensorBase& output)
{
    auto shapeTmp = output.GetShape();
    std::vector<size_t> shape { shapeTmp.begin(), shapeTmp.end() };

    if (output.GetDataType() == Base::TENSOR_DTYPE_FLOAT32) {
        cnpy::NpySave(outputFileName, (float*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_FLOAT16) {
        cnpy::NpySave(outputFileName, (aclFloat16*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_INT8) {
        cnpy::NpySave(outputFileName, (int8_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_INT32) {
        cnpy::NpySave(outputFileName, (int32_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_UINT8) {
        cnpy::NpySave(outputFileName, (uint8_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_INT16) {
        cnpy::NpySave(outputFileName, (int16_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_UINT16) {
        cnpy::NpySave(outputFileName, (uint16_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_UINT32) {
        cnpy::NpySave(outputFileName, (uint32_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_INT64) {
        cnpy::NpySave(outputFileName, (int64_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_UINT64) {
        cnpy::NpySave(outputFileName, (uint64_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_DOUBLE64) {
        cnpy::NpySave(outputFileName, (double*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_BOOL) {
        cnpy::NpySave(outputFileName, (bool*)output.GetBuffer(), shape);
    }
    return 0;
}


void FuncSave(ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue, int32_t deviceId, std::string outputDir)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    while (true)
    {
        auto item = saveQueue.pop();
        if (!item) {
            break;
        }
        for (auto &mem : *(item->memory)) {
            Base::MemoryHelper::Free(mem);
        }

        if (outputDir != "") {
            size_t n = item->outputs->size();
            for (size_t i = 0; i < n; i++) {
                std::string outputFileName = outputDir + RemoveSlash(
                    item->outputPrefix + "_" + item->outputNames->at(i) + "_" + std::to_string(i) + ".npy");
                if (TensotToNumpy(outputFileName, item->outputs->at(i))) {
                    ERROR_LOG("%s save failed\n", outputFileName.c_str());
                }
            }
        }
    }
}

void Execute(Arguments& arguments)
{
    std::string input = arguments["input"];
    std::vector<std::vector<std::string>> filesList {};
    CreateFilesList(filesList, input);

    std::shared_ptr<Base::SessionOptions> options = std::make_shared<Base::SessionOptions>();

    options->loop = stoi(arguments["loop"]);
    options->log_level = arguments["debug"] == "0" ? LOG_INFO_LEVEL : LOG_DEBUG_LEVEL;
    size_t deviceId = stoi(arguments["device"]);

    auto session = std::make_shared<Base::PyInferenceSession>(arguments["model"], deviceId, options);
    SetSession(session, arguments);

    ConcurrentQueue<std::shared_ptr<Feeds>> h2dQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> computeQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> d2hQueue;
    ConcurrentQueue<std::shared_ptr<Feeds>> saveQueue;

    std::vector<TimePointPair> h2dTs;
    std::vector<TimePointPair> computeTs;
    std::vector<TimePointPair> d2hTs;

    auto start = chr::steady_clock::now();

    std::thread h2dThread(FuncH2d, std::ref(h2dQueue), std::ref(computeQueue), deviceId, session, std::ref(h2dTs));
    std::thread computeThread(FuncCompute, std::ref(computeQueue), std::ref(d2hQueue),
                              deviceId, session, std::ref(computeTs));
    std::thread d2hThread(FuncD2h, std::ref(d2hQueue), std::ref(saveQueue), deviceId, std::ref(d2hTs));
    std::thread saveThread(FuncSave, std::ref(saveQueue), deviceId, arguments["output"]);
    FuncPrepare(deviceId, session, arguments["model"], options,
                filesList, h2dQueue, arguments["auto_set_dymshape_mode"] != "0");

    h2dThread.join();
    computeThread.join();
    d2hThread.join();
    saveThread.join();

    auto end = chr::steady_clock::now();
    auto e2ems = chr::duration_cast<chr::milliseconds>(end - start).count();
    INFO_LOG("End2End time: %ld ms\n", e2ems);

    PrintTimeWall("h2d", h2dTs);
    PrintTimeWall("compute", computeTs);
    PrintTimeWall("d2h", d2hTs);

    session->Finalize();
}


int main(int argc, char **argv) {
    Arguments arguments{{"model", ""}, {"input", ""}, {"output", ""}, {"loop", "1"}, {"debug", "0"}, {"warmup", "1"},
                    {"device", ""}, {"dymHW", ""}, {"dymDims", ""}, {"dymShape", ""}, {"display", "0"},
                    {"outputSize", ""}, {"auto_set_dymshape_mode", "0"}};

    ReadArgs(argc, argv, arguments);

    Execute(arguments);
}