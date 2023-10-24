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
    void AutoSetDym(std::shared_ptr<Feeds> feeds, std::string type, std::string inputName,
                    std::vector<size_t> &shape, bool last)
    {
        auto &target = type == "shape" ? feeds->autoDynamicShape : feeds->autoDynamicDims;
        target += Utils::CreateDynamicShapeDims(inputName, shape);
        if (!last) {
            target += ";";
        }
    }

    cnpy::NpyArray CreatePureInferArray(std::string fname, Base::TensorDesc inTensor)
    {
        size_t size = inTensor.realsize;
        cnpy::NpyArray arr = {};
        try {
            arr.dataHolder = std::make_shared<std::vector<char>>(size);
        } catch (exception &e) {
            throw std::runtime_error("Create pure data: make dataHolder failed");
        }
        arr.dataHolder = std::make_shared<std::vector<char>>(size);
        cnpy::DataUnion tmpTrans;
        srand(time(NULL));
        for (size_t i = 0; i < size; ++i) {
            if (fname == "pure_infer_data_zero") {
                tmpTrans.value = 0;
            } else if (fname == "pure_infer_data_random") {
                uint8_t min = 0;
                uint8_t max = UINT8_MAX - 1; // avoid float ±inf
                tmpTrans.value = (rand() % (max - min + 1)) + min;
            }
            arr.dataHolder->data()[i] = tmpTrans.bytes;
        }
        return arr;
    }

    void PrepareInputData(std::vector<std::string> &files, Base::PyInferenceSession* session,
        std::shared_ptr<Feeds> &feeds, bool autoDymShape,
        bool autoDymDims, const bool pure_infer, std::vector<std::string> &inputNames)
    {
        for (size_t i = 0; i < files.size(); i++) {
            if (pure_infer) {
                auto array = std::make_shared<cnpy::NpyArray>(CreatePureInferArray(files[i], session->GetInputs()[i]));
                feeds->arrayPtr->emplace_back(array);
            } else {
                if (Utils::TailContain(files[i], ".npy") || Utils::TailContain(files[i], ".NPY")) {
                    auto array = std::make_shared<cnpy::NpyArray>(cnpy::NpyLoad(files[i]));
                    feeds->arrayPtr->emplace_back(array);
                } else {
                    auto array = std::make_shared<cnpy::NpyArray>(cnpy::BinLoad(files[i]));
                    feeds->arrayPtr->emplace_back(array);
                }
            }

            feeds->inputs->emplace_back(feeds->arrayPtr->back()->Data<void>(), feeds->arrayPtr->back()->NumBytes());
            if (autoDymShape) {
                AutoSetDym(feeds, "shape", inputNames[i], feeds->arrayPtr->back()->shape, i == (files.size() - 1));
            }
            if (autoDymDims) {
                AutoSetDym(feeds, "dim", inputNames[i], feeds->arrayPtr->back()->shape, i == (files.size() - 1));
            }
        }
    }

    void FuncPrepare(ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue,
                     Base::PyInferenceSession* session,
                     std::vector<std::vector<std::string>> &infilesList,
                     std::shared_ptr<InferOptions> inferOption, size_t numThreads, size_t startIndex)
    {
        std::vector<std::string> inputNames {};
        std::vector<std::string> outputNames {};
        for (const auto &desc: session->GetInputs()) {
            inputNames.emplace_back(desc.name);
        }
        for (const auto &desc: session->GetOutputs()) {
            outputNames.emplace_back(desc.name);
        }
        size_t n = infilesList.size();
        for (size_t i = startIndex; i < n; i += numThreads) {
            auto &files = infilesList[i];
            auto feeds = std::make_shared<Feeds>();

            feeds->outputNames = std::make_shared<std::vector<std::string>>(outputNames);
            if (inferOption->outputDir != "") {
                for (auto tail : {".npy", ".bin", ".NPY", ".BIN", ""}) {
                    if (Utils::TailContain(files.front(), tail)) {
                        feeds->outputPrefix = Utils::GetPrefix(inferOption->outputDir, files.front(), tail);
                        break;
                    }
                }
            }
            feeds->inputs = std::make_shared<std::vector<Base::BaseTensor>>();
            feeds->arrayPtr = std::make_shared<std::vector<std::shared_ptr<cnpy::NpyArray>>>();
            PrepareInputData(files, session, feeds, inferOption->autoDymShape, inferOption->autoDymDims,
                             inferOption->pureInferMode, inputNames);
            h2dQueue.push(feeds);
        }
        h2dQueue.push(nullptr);
    }

    void FuncPrepareBaseTensor(ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue, uint32_t deviceId,
                               Base::PyInferenceSession* session,
                               std::vector<std::vector<Base::BaseTensor>>& inputsList,
                               std::vector<std::vector<std::vector<size_t>>>& shapesList, bool autoDymShape,
                               bool autoDymDims, std::vector<std::string>& outputNames)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(session->GetDeviceId(),
                                                                       session->GetContextIndex());
        if (ret != APP_ERR_OK) {
            throw std::runtime_error(GetError(ret));
        }
        std::vector<std::string> inputNames {};
        for (const auto &desc: session->GetInputs()) {
            inputNames.emplace_back(desc.name);
        }
        for (size_t i = 0; i < inputsList.size(); i++) {
            auto feeds = std::make_shared<Feeds>();
            feeds->inputs = std::make_shared<std::vector<Base::BaseTensor>>(inputsList[i]);
            feeds->outputNames = std::make_shared<std::vector<std::string>>(outputNames);
            for (size_t j = 0; j < inputNames.size(); j++) {
                if (autoDymShape) {
                    AutoSetDym(feeds, "shape", inputNames[j], shapesList[i][j], j == (inputNames.size() - 1));
                }
                if (autoDymDims) {
                    AutoSetDym(feeds, "dim", inputNames[j], shapesList[i][j], j == (inputNames.size() - 1));
                }
            }
            h2dQueue.push(feeds);
        }
        h2dQueue.push(nullptr);
    }

    void FuncH2d(ConcurrentQueue<std::shared_ptr<Feeds>> &h2dQueue,
                 ConcurrentQueue<std::shared_ptr<Feeds>> &computeQueue,
                 Base::PyInferenceSession* session)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(session->GetDeviceId(),
                                                                       session->GetContextIndex());
        if (ret != APP_ERR_OK) {
            throw std::runtime_error(GetError(ret));
        }
        while (true) {
            auto item = h2dQueue.pop();
            if (!item) {
                computeQueue.push(nullptr);
                break;
            }

            item->memory = std::make_shared<std::vector<Base::MemoryData>>();
            auto inputs = std::make_shared<std::vector<Base::BaseTensor>>();
            for (auto &info : *(item->inputs)) {
                Base::MemoryData mem = Base::CopyMemory2DeviceMemory(info.buf, info.size, session->GetDeviceId());
                item->memory->emplace_back(mem);
                Base::BaseTensor tensor(mem.ptrData, mem.size);
                inputs->emplace_back(tensor);
            }
            item->inputs = inputs;

            computeQueue.push(item);
        }
    }

    void FuncCompute(ConcurrentQueue<std::shared_ptr<Feeds>> &computeQueue,
                     ConcurrentQueue<std::shared_ptr<Feeds>> &d2hQueue,
                     Base::PyInferenceSession* session, InferSumaryInfo* summaryInfo)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(session->GetDeviceId(),
                                                                       session->GetContextIndex());
        if (ret != APP_ERR_OK) {
            throw std::runtime_error(GetError(ret));
        }
        while (true) {
            auto item = computeQueue.pop();
            if (!item) {
                d2hQueue.push(nullptr);
                if (summaryInfo != nullptr) {
                    *summaryInfo = session->GetSumaryInfo();
                }
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
                 ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue,
                 Base::PyInferenceSession* session)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(session->GetDeviceId(),
                                                                       session->GetContextIndex());
        if (ret != APP_ERR_OK) {
            throw std::runtime_error(GetError(ret));
        }
        while (true) {
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

    void SaveOutput(std::shared_ptr<Feeds> item, std::string outFmt, size_t index)
    {
        std::string outputFileName = item->outputPrefix + Utils::RemoveSlash(item->outputNames->at(index));
        if (outFmt == "NPY") {
            outputFileName += ".npy";
            if (Utils::TensorToNumpy(outputFileName, item->outputs->at(index)) == FAILED) {
                ERROR_LOG("%s save failed\n", outputFileName.c_str());
            }
        } else if (outFmt == "TXT") {
            outputFileName += ".txt";
            if (Utils::TensorToTxt(outputFileName, item->outputs->at(index)) == FAILED) {
                ERROR_LOG("%s save failed\n", outputFileName.c_str());
            }
        } else {
            outputFileName += ".bin";
            if (Utils::TensorToBin(outputFileName, item->outputs->at(index)) == FAILED) {
                ERROR_LOG("%s save failed\n", outputFileName.c_str());
            }
        }
    }

    void FuncSave(ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue, std::shared_ptr<InferOptions> inferOption)
    {
        while (true) {
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
                    SaveOutput(item, inferOption->outFmt, i);
                }
            }
        }
    }

    void FuncSaveTensorBase(ConcurrentQueue<std::shared_ptr<Feeds>> &saveQueue,
                            std::vector<std::vector<TensorBase>> &result, Base::PyInferenceSession* session)
    {
        APP_ERROR ret = Base::TensorContext::GetInstance()->SetContext(session->GetDeviceId(),
                                                                       session->GetContextIndex());
        if (ret != APP_ERR_OK) {
            throw std::runtime_error(GetError(ret));
        }

        while (true) {
            auto item = saveQueue.pop();
            if (!item) {
                break;
            }
            for (auto &mem : *(item->memory)) {
                Base::MemoryHelper::Free(mem);
            }
            result.emplace_back(*(item->outputs));
        }
    }
}