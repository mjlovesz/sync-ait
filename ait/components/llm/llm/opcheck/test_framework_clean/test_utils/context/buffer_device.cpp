/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "buffer_device.h"
#include <acl/acl_rt.h>
#include <atb/log.h>

namespace atb {
BufferDevice::BufferDevice(uint64_t bufferSize) : bufferSize_(bufferSize)
{
    ATB_LOG(INFO) << "BufferDevice::BufferDevice called, bufferSize:" << bufferSize;
    if (bufferSize_ > 0) {
        ATB_LOG(FATAL) << "BufferDevice::GetBuffer aclrtMalloc bufferSize:" << bufferSize_;
        int st = aclrtMalloc(static_cast<void **>(&buffer_), bufferSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (st != ACL_RT_SUCCESS) {
            ATB_LOG(FATAL) << "BufferDevice::GetBuffer aclrtMalloc fail, ret:" << st;
        }
    }
}

BufferDevice::~BufferDevice()
{
    Free();
}

void *BufferDevice::GetBuffer(uint64_t bufferSize)
{
    if (bufferSize <= bufferSize_) {
        ATB_LOG(INFO) << "BufferDevice::GetBuffer bufferSize:" << bufferSize << " <= bufferSize_:" << bufferSize_ <<
            ", not new device mem";
        return buffer_;
    }

    Free();

    ATB_LOG(FATAL) << "BufferDevice::GetBuffer aclrtMalloc bufferSize:" << bufferSize;
    int st = aclrtMalloc(static_cast<void **>(&buffer_), bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (st != ACL_RT_SUCCESS) {
        ATB_LOG(ERROR) << "BufferDevice::GetBuffer aclrtMalloc fail, ret:" << st;
        return nullptr;
    }
    ATB_LOG(INFO) << "BufferDevice::GetBuffer aclrtMalloc success, buffer:" << buffer_;
    bufferSize_ = bufferSize;
    return buffer_;
}

void BufferDevice::Free()
{
    if (buffer_) {
        ATB_LOG(INFO) << "BufferDevice::GetBuffer aclrtFree buffer:" << buffer_ << ", bufferSize:" <<
            bufferSize_;
        aclrtFree(buffer_);
        buffer_ = nullptr;
        bufferSize_ = 0;
    }
}
} // namespace atb