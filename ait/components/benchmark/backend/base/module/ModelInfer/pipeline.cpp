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

#include "Base/ModelInfer/pipeline.h"

namespace Base {
    void FuncPrepare(ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue, uint32_t deviceId, Base::PyInferenceSession* session,
                     std::vector<std::vector<std::string>> &infilesList, bool autoDymShape, bool autoDymDims,
                     const std::string &outputDir)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
        if (ret != APP_ERR_OK) {
            throw std::runtime_error(GetError(ret));
        }
        std::vector<std::string> inputNames {};
        std::vector<std::string> outputNames {};
        for (const auto &desc: session->GetInputs()) {
            inputNames.emplace_back(desc.name);
        }
        for (const auto &desc: session->GetOutputs()) {
            outputNames.emplace_back(desc.name);
        }
        if (outputDir != "") {
            if (fs::is_symlink(outputDir)) {
                fs::remove(outputDir);
            }
            fs::create_directories(outputDir);
        }

        for (auto &files : infilesList) {
            auto feeds = std::make_shared<Feeds>();

            feeds->outputNames = std::make_shared<std::vector<std::string>>(outputNames);
            if (outputDir != "") {
                for (auto tail : {".npy", ".NPY", ".bin", ".BIN", ""}) {
                    if (Utils::TailContain(files.front(), tail)) {
                        feeds->outputPrefix = Utils::GetPrefix(outputDir, files.front(), tail);
                    }
                }
            }
            feeds->inputs = std::make_shared<std::vector<Base::BaseTensor>>();
            feeds->arrayPtr = std::make_shared<std::vector<std::shared_ptr<cnpy::NpyArray>>>();

            for (size_t i = 0; i < files.size(); i++) {
                if (Utils::TailContain(files[i], ".npy") || Utils::TailContain(files[i], ".NPY")) {
                    auto array = std::make_shared<cnpy::NpyArray>(cnpy::NpyLoad(files[i]));
                    feeds->arrayPtr->emplace_back(array);
                } else {
                    auto array = std::make_shared<cnpy::NpyArray>(cnpy::BinLoad(files[i]));
                    feeds->arrayPtr->emplace_back(array);
                }
                feeds->inputs->emplace_back(feeds->arrayPtr->back()->Data<void>(), feeds->arrayPtr->back()->NumBytes());
                if (autoDymShape) {
                    feeds->autoDynamicShape += Utils::CreateDynamicShapeDims(inputNames[i], feeds->arrayPtr->back()->shape);
                    if (i != files.size() - 1) {
                        feeds->autoDynamicShape += ";";
                    }
                }
                if (autoDymDims) {
                    feeds->autoDynamicDims += Utils::CreateDynamicShapeDims(inputNames[i], feeds->arrayPtr->back()->shape);
                    if (i != files.size() - 1) {
                        feeds->autoDynamicDims += ";";
                    }
                }
            }
            h2dQueue.push(feeds);
        }
        h2dQueue.push(nullptr);
    }

    void FuncH2d(ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue,
                    ConcurrentQueue<std::shared_ptr<Feeds>> &computeQueue, uint32_t deviceId)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
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

            item->memory = std::make_shared<std::vector<Base::MemoryData>>();
            auto inputs = std::make_shared<std::vector<Base::BaseTensor>>();
            for (auto &info : *(item->inputs)) {
                Base::MemoryData mem = Base::CopyMemory2DeviceMemory(info.buf, info.size, deviceId);
                item->memory->emplace_back(mem);
                Base::BaseTensor tensor(mem.ptrData, mem.size);
                inputs->emplace_back(tensor);
            }
            item->inputs = inputs;

            computeQueue.push(item);
        }
    }

    void FuncCompute(ConcurrentQueue<std::shared_ptr<Feeds>> &computeQueue,
                        ConcurrentQueue<std::shared_ptr<Feeds>> &d2hQueue, uint32_t deviceId,
                        Base::PyInferenceSession* session)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
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

            if (item->autoDynamicShape != "") {
                session->SetDynamicShape(item->autoDynamicShape);
            }
            if (item->autoDynamicDims != "") {
                session->SetDynamicDims(item->autoDynamicDims);
            }

            item->outputs = std::make_shared<std::vector<Base::TensorBase>>();
            session->OnlyInfer(*(item->inputs), *(item->outputNames), *(item->outputs));

            d2hQueue.push(item);
        }
    }

    void FuncD2h(ConcurrentQueue<std::shared_ptr<Feeds>> &d2hQueue,
                    ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue, uint32_t deviceId)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
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

            for (auto &output : *(item->outputs)) {
                Base::TensorToHost(output);
            }

            saveQueue.push(item);
        }
    }

    void FuncSave(ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue, uint32_t deviceId, std::string outFmt)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(deviceId);
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

            if (item->outputPrefix != "") {
                size_t n = item->outputs->size();
                for (size_t i = 0; i < n; i++) {
                    std::string outputFileName = item->outputPrefix + Utils::RemoveSlash(item->outputNames->at(i));
                    if (outFmt == "NPY") {
                        outputFileName += ".npy";
                        if (Utils::TensorToNumpy(outputFileName, item->outputs->at(i)) == FAILED) {
                            ERROR_LOG("%s save failed\n", outputFileName.c_str());
                        }
                    } else if (outFmt == "TXT") {
                        outputFileName += ".txt";
                        if (Utils::TensorToTxt(outputFileName, item->outputs->at(i)) == FAILED) {
                            ERROR_LOG("%s save failed\n", outputFileName.c_str());
                        }
                    } else {
                        outputFileName += ".bin";
                        if (Utils::TensorToBin(outputFileName, item->outputs->at(i)) == FAILED) {
                            ERROR_LOG("%s save failed\n", outputFileName.c_str());
                        }
                    }
                }
            }
        }
    }
}