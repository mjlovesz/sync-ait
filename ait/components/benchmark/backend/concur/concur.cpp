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
#include "PyTensor/PyTensor.h"
#include "cnpy.h"
#include "utils.h"

#include "Base/ModelInfer/SessionOptions.h"
#include "Base/ModelInfer/ModelInferenceProcessor.h"
#include "PyInferenceSession/PyInferenceSession.h"

namespace chr = std::chrono;
using TimePointPair = std::pair<chr::steady_clock::time_point, chr::steady_clock::time_point>;
using Arguments = std::unordered_map<std::string, std::string>;

struct Feeds{
    std::shared_ptr<std::vector<std::string>> _output_names = nullptr;
    std::shared_ptr<std::vector<Base::BaseTensor>> _inputs = nullptr;
    std::shared_ptr<std::vector<Base::TensorBase>> _outputs = nullptr;
    std::shared_ptr<std::vector<Base::MemoryData>> _memory = nullptr;
    std::shared_ptr<std::vector<std::shared_ptr<cnpy::NpyArray>>> _arrayptr = nullptr;
    std::string _autoDynamicShape = "";
    std::string _output_prefix = "";
};



template <typename T>
class ConcurrentQueue {
    public:
    ConcurrentQueue(int depth = 3): _depth(depth) {}

    T pop() {
        std::unique_lock<std::mutex> mlock(_mtx);
        while (_queue.empty()) {
            _cond_var.wait(mlock);
        }
        auto val = _queue.front();
        _queue.pop();
        mlock.unlock();
        _cond_var.notify_one();
        return val;
    }

    void push (const T &item) {
        std::unique_lock<std::mutex> mlock(_mtx);
        while (_queue.size() >= _depth) {
            _cond_var.wait(mlock);
        }
        _queue.push(item);
        mlock.unlock();
        _cond_var.notify_one();
    }

    private:
    std::mutex _mtx;
    std::queue<T> _queue;
    std::condition_variable _cond_var;
    int _depth;
};

void setSession(std::shared_ptr<Base::PyInferenceSession> session, Arguments& arguments)
{
    if (arguments["dymHW"] != "") {
        auto dymhw = split(arguments["dymHW"], ',');
        session->SetDynamicHW(stoi(dymhw[0]), stoi(dymhw[1]));
    }
    if (arguments["dymDims"] != "") {
        session->SetDynamicDims(arguments["dymDims"]);
    }
    if (arguments["dymShape"] != "") {
        session ->SetDynamicShape(arguments["dymShape"]);
    }
    if (arguments["outputSize"] != "") {
        auto outputSize = strVecToNumVec(split(arguments["outputSize"], ','));
        session->SetCustomOutTensorsSize(outputSize);
    }
}

void func_prepare(int32_t deviceId, std::shared_ptr<Base::PyInferenceSession> session, std::string modelPath,
                  std::shared_ptr<Base::SessionOptions> options, std::vector<std::vector<std::string>> &filesList,
                  ConcurrentQueue<std::shared_ptr<Feeds>> &h2d_queue, bool autoDymShape)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    std::vector<std::string> input_names{};
    if (autoDymShape) {
        auto intensor_desc = session->GetInputs();
        for (auto &desc: intensor_desc) {
            input_names.push_back(desc.name);
        }
    }
    
    for (auto &files: filesList) {
        auto output_names = std::make_shared<std::vector<std::string>>();
        for (const auto &desc: session->GetOutputs()) {
            output_names->push_back(desc.name);
        }
        auto inputs = std::make_shared<std::vector<Base::BaseTensor>>();
        auto arrayptr =std::make_shared<std::vector<std::shared_ptr<cnpy::NpyArray>>>();
        std::string autoDynamicShape{};
        for (size_t i = 0; i < files.size(); i++) {
            auto array = std::make_shared<cnpy::NpyArray>(cnpy::npy_load(files[i]));
            arrayptr->emplace_back(array);
            inputs->emplace_back(array->data<void>(), array->num_bytes());
            if (i != files.size()-1) {
                autoDynamicShape += ";";
            }
        }
        auto output_prefix = getPrefix(files.front());
        auto feeds = std::make_shared<Feeds>();
        feeds->_autoDynamicShape = autoDynamicShape;
        feeds->_arrayptr = arrayptr;
        feeds->_output_names = output_names;
        feeds->_inputs = inputs;
        feeds->_output_prefix = output_prefix;
        h2d_queue.push(feeds);
    }
    h2d_queue.push(nullptr);
}

void func_h2d(ConcurrentQueue<std::shared_ptr<Feeds>> &h2d_queue,
              ConcurrentQueue<std::shared_ptr<Feeds>> &compute_queue, int32_t deviceId,
              std::shared_ptr<Base::PyInferenceSession> session,
              std::vector<TimePointPair>& timestamps)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }
    while (true)
    {
        auto item = h2d_queue.pop();
        if (!item) {
            compute_queue.push(nullptr);
            break;
        }
        auto start = chr::steady_clock::now();

        item->_memory = std::make_shared<std::vector<Base::MemoryData>>();
        auto inputs = std::make_shared<std::vector<Base::BaseTensor>>();
        for (auto &info: *(item->_inputs)) {
            Base::MemoryData mem = Base::CopyMemory2DeviceMemory(info.buf, info.size, deviceId);
            item->_memory->push_back(mem);
            Base::BaseTensor tensor(mem.ptrData, mem.size);
            inputs->push_back(tensor);
        }
        item->_inputs = inputs;
        auto end = chr::steady_clock::now();
        timestamps.emplace_back(start, end);

        compute_queue.push(item);
    }
}

void func_compute(ConcurrentQueue<std::shared_ptr<Feeds>> &compute_queue,
                 ConcurrentQueue<std::shared_ptr<Feeds>> &d2h_queue, int32_t deviceId,
                 std::shared_ptr<Base::PyInferenceSession> session,
                 std::vector<TimePointPair>& timestamps)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    while (true)
    {
        auto item = compute_queue.pop();
        if (!item) {
            d2h_queue.push(nullptr);
            break;
        }
        auto start = chr::steady_clock::now();
        
        if (item->_autoDynamicShape != "") {
            session->SetDynamicShape(item->_autoDynamicShape);
        }
        auto outputs = std::make_shared<std::vector<Base::TensorBase>>();
        Base::PyInferenceSession::PureInfer(*(item->_inputs), *(item->_output_names), *outputs);
        item->_outputs = outputs;

        auto end = chr::steady_clock::now();
        timestamps.emplace_back(start, end);

        d2h_queue.push(item);
    }
    
}

void func_d2h(ConcurrentQueue<std::shared_ptr<Feeds>> &d2h_queue,
                 ConcurrentQueue<std::shared_ptr<Feeds>> &save_queue, int32_t deviceId,
                 std::vector<TimePointPair>& timestamps)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    while (true)
    {
        auto item = d2h_queue.pop();
        if (!item) {
            save_queue.push(nullptr);
            break;
        }
        auto start = chr::steady_clock::now();
        
        for (auto &output: *(item->_outputs)) {
            Base::TensorToHost(output);
        }

        auto end = chr::steady_clock::now();
        timestamps.emplace_back(start, end);

        save_queue.push(item);
    }
    
}

int tensotToNumpy(std::string output_filename, Base::TensorBase& output)
{
    auto shape_tmp = output.GetShape();
    std::vector<size_t> shape{shape_tmp.begin(), shape_tmp.end()};

    if (output.GetDataType() == Base::TENSOR_DTYPE_FLOAT32) {
        cnpy::npy_save(output_filename, (float*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_FLOAT16) {
        cnpy::npy_save(output_filename, (short*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_INT8) {
        cnpy::npy_save(output_filename, (int8_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_INT32) {
        cnpy::npy_save(output_filename, (int32_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_UINT8) {
        cnpy::npy_save(output_filename, (uint8_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_INT16) {
        cnpy::npy_save(output_filename, (int16_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_UINT16) {
        cnpy::npy_save(output_filename, (uint16_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_UINT32) {
        cnpy::npy_save(output_filename, (uint32_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_INT64) {
        cnpy::npy_save(output_filename, (int64_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_UINT64) {
        cnpy::npy_save(output_filename, (uint64_t*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_DOUBLE64) {
        cnpy::npy_save(output_filename, (double*)output.GetBuffer(), shape);
    } else if (output.GetDataType() == Base::TENSOR_DTYPE_BOOL) {
        cnpy::npy_save(output_filename, (bool*)output.GetBuffer(), shape);
    }
    return 0;
}


void func_save(ConcurrentQueue<std::shared_ptr<Feeds>> &save_queue, int32_t deviceId, std::string output_dir)
{
    APP_ERROR ret;
    ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
    if (ret != APP_ERR_OK) {
        throw std::runtime_error(GetError(ret));
    }

    while (true)
    {
        auto item = save_queue.pop();
        if (!item) {
            break;
        }
        for (auto &mem: *(item->_memory)) {
            Base::MemoryHelper::Free(mem);
        }

        if (output_dir != "") {
            size_t n = item->_outputs->size();
            for (size_t i = 0; i < n; i++) {
                std::string output_filename = output_dir + removeSlash(
                    item->_output_prefix + "_" + item->_output_names->at(i) + "_" + std::to_string(i) + ".npy");
                if (tensotToNumpy(output_filename, item->_outputs->at(i))) {
                    std::cout << output_filename << "save failed" << endl;
                }
            }
        }
    }
    
}

void Execute(Arguments& arguments)
{
    std::string input = arguments["input"];
    std::vector<std::vector<std::string>> filesList{};
    createFilesList(filesList, input);

    std::shared_ptr<Base::SessionOptions> options = std::make_shared<Base::SessionOptions>();

    options->loop = stoi(arguments["loop"]);
    options->log_level = arguments["debug"] == "0" ? 2 : 1;
    size_t deviceId = stoi(arguments["device"]);
    
    auto session = std::make_shared<Base::PyInferenceSession>(arguments["model"], deviceId, options);
    setSession(session, arguments);

    ConcurrentQueue<std::shared_ptr<Feeds>> h2d_queue;
    ConcurrentQueue<std::shared_ptr<Feeds>> compute_queue;
    ConcurrentQueue<std::shared_ptr<Feeds>> d2h_queue;
    ConcurrentQueue<std::shared_ptr<Feeds>> save_queue;

    std::vector<TimePointPair> h2d_ts;
    std::vector<TimePointPair> compute_ts;
    std::vector<TimePointPair> d2h_ts;
    

    auto start = chr::steady_clock::now();

    std::thread h2dThread(func_h2d, std::ref(h2d_queue), std::ref(compute_queue), deviceId, session, std::ref(h2d_ts));
    std::thread computeThread(func_compute, std::ref(compute_queue), std::ref(d2h_queue), deviceId, session, std::ref(compute_ts));
    std::thread d2hThread(func_d2h, std::ref(d2h_queue), std::ref(save_queue), deviceId, std::ref(h2d_ts));
    std::thread saveThread(func_save, std::ref(save_queue), deviceId, arguments["output"]);
    
    func_prepare(deviceId, session, arguments["model"], options, filesList, h2d_queue, arguments["auto_set_dymshape_mode"] != "0");

    h2dThread.join();
    computeThread.join();
    d2hThread.join();
    saveThread.join();

    auto end = chr::steady_clock::now();
    auto e2e_ms = chr::duration_cast<chr::milliseconds>(end - start).count();
    std::cout << "End2End time: " << e2e_ms << "ms" << std::endl;

    printTimeWall("h2d", h2d_ts);
    printTimeWall("compute", compute_ts);
    printTimeWall("d2h", d2h_ts);

    session->Finalize();


}


int main(int argc, char **argv) {

    Arguments arguments{{"model", ""}, {"input", ""}, {"output", ""}, {"loop", "1"}, {"debug", "0"}, {"warmup", "1"},
                    {"device", ""}, {"dymHW", ""}, {"dymDims", ""}, {"dymShape", ""}, {"display", "0"},
                    {"outputSize", ""}, {"auto_set_dymshape_mode", "0"}};
    
    readArgs(argc, argv, arguments);

    Execute(arguments);
}