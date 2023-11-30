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


#include "atb_probe.h"
#include "binfile.h"
static std::vector<std::string> SplitString(const std::string &ss, const char &tar)
{
    std::vector<std::string> tokens;
    std::stringstream input(ss);
    std::string token;
    while (std::getline(input, token, tar))
    {
        tokens.push_back(token);
    }

    return tokens;

}


bool atb::Probe::IsTensorNeedSave(const std::vector<int64_t> &ids, std::string &optype)
{
    const char *vid = std::getenv("ATB_SAVE_TENSOR_IDS"); // 应该是20_1_9,1_23,5_29_1
    const char *tid = std::getenv("ATB_SAVE_TENSOR_RUNNER"); // 应该是LinearOps，SelfAttention
    if (!vid && !tid)
    {
        return true;
    }
    // 先用逗号分隔vid和tid
    std::vector<std::string> splitVid = SplitString(vid, ',');
    std::vector<std::string> splitTid = SplitString(tid, ',');
    std::string query = "";
    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (i)
        {
            query += "_" + std::to_string(ids[i]);
        }
        else 
        {
            query += std::to_string(ids[i]);
        }
    }
    for (auto &indice : splitVid)
    {
        if (indice == query) {
            return true;
        }
    }
    for (auto &indice : splitTid)
    {
        if (indice == optype) {
            return true;
        }
    }
    return false;
}


bool atb::Probe::IsSaveTensorData()
{
    const char* saveTensor = std::getenv("ATB_SAVE_TENSOR");
    if (saveTensor == "1")
    {
        return true;
    }
    return false;
}


bool atb::Probe::IsSaveTensorDesc()
{
    return true;
}


bool atb::Probe::IsExecuteCountInRange(const uint64_t executeCount)
{
    const char* saveTensorRange = std::getenv("ATB_SAVE_TENSOR_RANGE");
    std::vector<std::string> saveTensorRan = SplitString(saveTensorRange, ',');
    for (size_t i = 1; i < saveTensorRan.size(); i += 2) {
        uint64_t left = stoi(saveTensorRan[i - 1]), right = stoi(saveTensorRan[i]);
        if (executeCount <= right && executeCount >= left) {
            return true;
        }
    }
    return false;
}


bool atb::Probe::IsSaveTensorBefore()
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    int value = std::stoi(saveTensorTime);
    if (value == 0 || value == 2) {
        return true;
    }
    return false;
}


bool atb::Probe::IsSaveTensorAfter()
{
    const char* saveTensorTime = std::getenv("ATB_SAVE_TENSOR_TIME");
    int value = std::stoi(saveTensorTime);
    if (value == 1 || value == 2) {
        return true;
    }
    return false;
}


void atb::Probe::SaveTensor(const std::string &format, const std::string &dtype,
        const std::string &dims, const void *hostData, uint64_t dataSize,
        const std::string &filePath)
{   int flag = std::stoi(std::getenv("LOG_TO_STDOUT"));
    if (!hostData)
    {   
        if (flag) {
            std::cout << "hostData is None." << std::endl;
        }
        return;
    }
    BinFile binFile;
    binFile.AddAttr("format", format);
    binFile.AddAttr("dtype", dtype);
    binFile.AddAttr("dims", dims);
    binFile.AddObject("data", hostData, dataSize);
    binFile.Write(filePath);

}


void atb::Probe::SaveTiling(const uint8_t* data, uint64_t dataSize, const std::string &filePath)
{
    std::ofstream outfile(filePath, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<const char*>(data), dataSize);
        outfile.close();
        std::cout << "Data written to file successfully!" << std::endl;
    } else {
        std::cout << "Unable to open file!" << std::endl;
    }
}


bool atb::Probe::IsSaveTiling()
{
    const char* isSaveTiling = std::getenv("ATB_SAVE_TILING");
    if (isSaveTiling == "1") {
        return true;
    }
    return false;
}