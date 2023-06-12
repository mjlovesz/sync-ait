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
using TimePointPair = std::pair<chr::steady_clock::time, chr::steady_clock::time_point>;
using Arguments = std::unordered_map<std::string, std::string>;

Arguments arguments{{"model", ""}, {"input", ""}, {"output", ""}, {"loop", "1"}, {"debug", "0"}, {"warmup", "1"},
                    {"device", ""}, {"dymHW", ""}, {"dymDims", ""}, {"dymShape", ""}, {"display", "0"},
                    {"outputSize", ""}, {"auto_set_dymshape_mode", "0"}};
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






int main(int argc, char **argv) {

}