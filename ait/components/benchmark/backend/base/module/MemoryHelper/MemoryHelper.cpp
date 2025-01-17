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

#include "Base/MemoryHelper/MemoryHelper.h"
#include <sys/time.h>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "Base/Log/Log.h"

namespace Base {
using MemeoryDataFreeFuncPointer = APP_ERROR (*)(void*);

APP_ERROR FreeFuncDelete(void* ptr)
{
    delete[] (int8_t*)ptr;
    return APP_ERR_OK;
}

APP_ERROR FreeFuncCFree(void* ptr)
{
    free(ptr);
    return APP_ERR_OK;
}

static MemorySummary g_MemorySummary;

struct MemorySummary* GetMemorySummaryPtr()
{
    return &g_MemorySummary;
}

void MemoryHelper::LogErrorInfo()
{
    LOG_ERROR << GetError(APP_ERR_ACL_BAD_ALLOC)<< "The module type is not defined.";
    return;
}

APP_ERROR MemoryHelper::specificMalloc(MemoryData& data)
{
    APP_ERROR ret = APP_ERR_OK;
    switch (data.type) {
        case MemoryData::MEMORY_HOST:
            ret = aclrtMallocHost(&(data.ptrData), data.size);
            data.free = aclrtFreeHost;
            if (ret != APP_ERR_OK) {
                cout << aclGetRecentErrMsg() << endl;
            }
            break;
        case MemoryData::MEMORY_DEVICE:
            ret = aclrtMalloc(&(data.ptrData), data.size, ACL_MEM_MALLOC_HUGE_FIRST);
            data.free = aclrtFree;
            if (ret != APP_ERR_OK) {
                cout << aclGetRecentErrMsg() << endl;
            }
            break;
        case MemoryData::MEMORY_DVPP:
            ret = acldvppMalloc(&(data.ptrData), data.size);
            data.free = acldvppFree;
            if (ret != APP_ERR_OK) {
                cout << aclGetRecentErrMsg() << endl;
            }
            break;
        case MemoryData::MEMORY_HOST_MALLOC:
            data.ptrData = malloc(data.size);
            if (data.ptrData == nullptr) {
                ret = APP_ERR_ACL_BAD_ALLOC;
            } else {
                ret = APP_ERR_OK;
            }
            data.free = (MemeoryDataFreeFuncPointer)FreeFuncCFree;
            break;
        case MemoryData::MEMORY_HOST_NEW:
            data.ptrData = (void*)(new int8_t[data.size]);
            if (data.ptrData == nullptr) {
                ret = APP_ERR_ACL_BAD_ALLOC;
            } else {
                ret = APP_ERR_OK;
            }
            data.free = (MemeoryDataFreeFuncPointer)FreeFuncDelete;
            break;
        default:
            LogErrorInfo();
            return APP_ERR_ACL_BAD_ALLOC;
    }
    return ret;
}

APP_ERROR MemoryHelper::Malloc(MemoryData& data)
{
    APP_ERROR ret = APP_ERR_OK;
    if (data.size == 0) {
        data.ptrData = nullptr;
        return APP_ERR_OK;
    }
    ret = specificMalloc(data);
    if (ret != APP_ERR_OK) {
        LOG_ERROR << GetError(ret) << "Malloc ptrData failed.";
        data.ptrData = nullptr;
        return APP_ERR_ACL_BAD_ALLOC;
    }
    return ret;
}

APP_ERROR MemoryHelper::Free(MemoryData& data)
{
    if (data.size == 0 && data.ptrData == nullptr) {
        return APP_ERR_OK;
    }
    if (data.ptrData == nullptr) {
        LOG_ERROR << GetError(APP_ERR_COMM_INVALID_POINTER)
                 << "Free failed, ptrData is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    APP_ERROR ret = APP_ERR_OK;
    int8_t *ptrData = nullptr;
    switch (data.type) {
        case MemoryData::MEMORY_HOST:
            ret = aclrtFreeHost(data.ptrData);
            if (ret != APP_ERR_OK) {
                cout << aclGetRecentErrMsg() << endl;
            }
            break;
        case MemoryData::MEMORY_DEVICE:
            ret = aclrtFree(data.ptrData);
            if (ret != APP_ERR_OK) {
                cout << aclGetRecentErrMsg() << endl;
            }
            break;
        case MemoryData::MEMORY_DVPP:
            ret = acldvppFree(data.ptrData);
            if (ret != APP_ERR_OK) {
                cout << aclGetRecentErrMsg() << endl;
            }
            break;
        case MemoryData::MEMORY_HOST_MALLOC:
            free(data.ptrData);
            ret = APP_ERR_OK;
            break;
        case MemoryData::MEMORY_HOST_NEW:
            ptrData = (int8_t*)data.ptrData;
            delete[] ptrData;
            ret = APP_ERR_OK;
            break;
        default:
            LOG_ERROR << GetError(APP_ERR_ACL_BAD_FREE)
                     << "Free failed, the module type is not defined, data type:" << data.type;
            return APP_ERR_ACL_BAD_FREE;
    }
    if (ret != APP_ERR_OK) {
        LOG_ERROR << GetError(ret) << "Free ptrData failed.";
        return APP_ERR_ACL_BAD_FREE;
    }
    data.ptrData = nullptr;
    return ret;
}

APP_ERROR MemoryHelper::Memset(MemoryData& data, int32_t value, size_t count)
{
    if (data.ptrData == nullptr) {
        LOG_ERROR << GetError(APP_ERR_COMM_INVALID_POINTER)
                 << "Memset failed, ptrData is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    APP_ERROR ret = aclrtMemset(data.ptrData, data.size, value, count);
    if (ret != APP_ERR_OK) {
        cout << aclGetRecentErrMsg() << endl;
        LOG_ERROR << GetError(ret) << "Memset ptrData failed.";
    }
    return ret;
}

void AddCostTime(float time, string type)
{
    std::lock_guard<std::mutex> lock(g_MemorySummary.mtx_);
    if (type == "h2d") {
        g_MemorySummary.H2DTimeList.push_back(time);
    } else if (type == "d2h") {
        g_MemorySummary.D2HTimeList.push_back(time);
    }
}

APP_ERROR MemoryHelper::Memcpy(MemoryData& dest, const MemoryData& src, size_t count)
{
    if (dest.size == 0 && src.size == 0) {
        return APP_ERR_OK;
    }
    if (dest.ptrData == nullptr || src.ptrData == nullptr) {
        LOG_ERROR << GetError(APP_ERR_COMM_INVALID_POINTER)
                 << "Memcpy failed, ptrData is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }
    APP_ERROR ret = APP_ERR_OK;
    struct timeval start = { 0 };
    struct timeval end = { 0 };
    float costTime;
    const float sec_to_usec = 1000.0;
    if (IsDeviceToHost(dest, src)) {
        gettimeofday(&start, nullptr);
        ret = aclrtMemcpy(dest.ptrData, dest.size, src.ptrData, count, ACL_MEMCPY_DEVICE_TO_HOST);
        gettimeofday(&end, nullptr);
        costTime = sec_to_usec * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / sec_to_usec;
        AddCostTime(costTime, "d2h");
    } else if (IsHostToHost(dest, src)) {
        ret = aclrtMemcpy(dest.ptrData, dest.size, src.ptrData, count, ACL_MEMCPY_HOST_TO_HOST);
    } else if (IsDeviceToDevice(dest, src)) {
        ret = aclrtMemcpy(dest.ptrData, dest.size, src.ptrData, count, ACL_MEMCPY_DEVICE_TO_DEVICE);
    } else if (IsHostToDevice(dest, src)) {
        gettimeofday(&start, nullptr);
        ret = aclrtMemcpy(dest.ptrData, dest.size, src.ptrData, count, ACL_MEMCPY_HOST_TO_DEVICE);
        gettimeofday(&end, nullptr);
        costTime = sec_to_usec * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / sec_to_usec;
        AddCostTime(costTime, "h2d");
    }
    if (ret != APP_ERR_OK) {
        cout << aclGetRecentErrMsg() << endl;
        LOG_ERROR << GetError(ret) << "Memcpy ptrData failed.";
        return APP_ERR_ACL_BAD_COPY;
    }
    return ret;
}

APP_ERROR MemoryHelper::MxbsMallocAndCopy(MemoryData& dest, const MemoryData& src)
{
    if (src.ptrData == nullptr) {
        LOG_ERROR << GetError(APP_ERR_COMM_INVALID_POINTER)
                 << "Memcpy failed, ptrData of src is nullptr.";
        return APP_ERR_COMM_INVALID_POINTER;
    }

    APP_ERROR ret = MemoryHelper::Malloc(dest);
    if (ret != APP_ERR_OK) {
        LOG_ERROR << GetError(ret) << "MxbsMallocAndCopy function malloc ptrData failed.";
        return ret;
    }

    ret = MemoryHelper::Memcpy(dest, src, src.size);
    if (ret != APP_ERR_OK) {
        LOG_ERROR << GetError(ret) << "MxbsMallocAndCopy function memcpy failed.";
        ret = dest.free(dest.ptrData);
        if (ret != APP_ERR_OK) {
            LOG_ERROR << GetError(ret) << "MxbsMallocAndCopy function free failed.";
        }
        dest.ptrData = nullptr;
        return APP_ERR_ACL_BAD_COPY;
    }
    return ret;
}

bool MemoryHelper::IsHostToDevice(const MemoryData& dest, const MemoryData& src)
{
    return (dest.type == MemoryData::MEMORY_DEVICE || dest.type == MemoryData::MEMORY_DVPP) &&
        (src.type == MemoryData::MEMORY_HOST || src.type == MemoryData::MEMORY_HOST_MALLOC ||
        src.type == MemoryData::MEMORY_HOST_NEW);
}

bool MemoryHelper::IsDeviceToDevice(const MemoryData& dest, const MemoryData& src)
{
    return (dest.type == MemoryData::MEMORY_DEVICE || dest.type == MemoryData::MEMORY_DVPP) &&
        (src.type == MemoryData::MEMORY_DEVICE || src.type == MemoryData::MEMORY_DVPP);
}

bool MemoryHelper::IsHostToHost(const MemoryData& dest, const MemoryData& src)
{
    return (dest.type == MemoryData::MEMORY_HOST || dest.type == MemoryData::MEMORY_HOST_MALLOC ||
        dest.type == MemoryData::MEMORY_HOST_NEW) &&
        (src.type == MemoryData::MEMORY_HOST || src.type == MemoryData::MEMORY_HOST_MALLOC ||
        src.type == MemoryData::MEMORY_HOST_NEW);
}

bool MemoryHelper::IsDeviceToHost(const MemoryData& dest, const MemoryData& src)
{
    return (dest.type == MemoryData::MEMORY_HOST || dest.type == MemoryData::MEMORY_HOST_MALLOC ||
        dest.type == MemoryData::MEMORY_HOST_NEW) &&
        (src.type == MemoryData::MEMORY_DEVICE || src.type == MemoryData::MEMORY_DVPP);
}

APP_ERROR MemoryHelper::MxbsMalloc(MemoryData& data)
{
    return MemoryHelper::Malloc(data);
}

APP_ERROR MemoryHelper::MxbsFree(MemoryData& data)
{
    return MemoryHelper::Free(data);
}

APP_ERROR MemoryHelper::MxbsMemset(MemoryData& data, int32_t value, size_t count)
{
    return MemoryHelper::Memset(data, value, count);
}

APP_ERROR MemoryHelper::MxbsMemcpy(MemoryData& dest, const MemoryData& src, size_t count)
{
    return MemoryHelper::Memcpy(dest, src, count);
}
}  // namespace Base
