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

#ifndef _PIPELINE_H
#define _PIPELINE_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "PyInferenceSession/PyInferenceSession.h"
#include "PyTensor/PyTensor.h"
#include "Base/Tensor/TensorBase/TensorBase.h"
#include "Base/ModelInfer/ModelInferenceProcessor.h"
#include "Base/Tensor/TensorContext/TensorContext.h"
#include "Base/ModelInfer/cnpy.h"
#include "Base/ModelInfer/utils.h"

using Arguments = std::unordered_map<std::string, std::string>;

struct Feeds {
    std::shared_ptr<std::vector<std::string>> outputNames = nullptr;
    std::shared_ptr<std::vector<Base::BaseTensor>> inputs = nullptr;
    std::shared_ptr<std::vector<Base::TensorBase>> outputs = nullptr;
    std::shared_ptr<std::vector<Base::MemoryData>> memory = nullptr;
    std::shared_ptr<std::vector<std::shared_ptr<cnpy::NpyArray>>> arrayPtr = nullptr;
    std::string autoDynamicShape = "";
    std::string autoDynamicDims = "";
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


namespace Base {
    void FuncPrepare(ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue, uint32_t deviceId, Base::PyInferenceSession* session,
                     std::vector<std::vector<std::string>> &infilesList, bool autoDymShape, bool autoDymDims,
                     const std::string &outputDir);

    void FuncH2d(ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue,
                 ConcurrentQueue<std::shared_ptr<Feeds>> &computeQueue, uint32_t deviceId);

    void FuncCompute(ConcurrentQueue<std::shared_ptr<Feeds>> &computeQueue,
                     ConcurrentQueue<std::shared_ptr<Feeds>> &d2hQueue, uint32_t deviceId,
                     Base::PyInferenceSession* session);

    void FuncD2h(ConcurrentQueue<std::shared_ptr<Feeds>> &d2hQueue,
                 ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue, uint32_t deviceId);

    void FuncSave(ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue, uint32_t deviceId);
}


#endif