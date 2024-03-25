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
#ifndef BUFFER_DEVICE_H
#define BUFFER_DEVICE_H
#include "buffer_base.h"

namespace atb {
class BufferDevice : public BufferBase {
public:
    explicit BufferDevice(uint64_t bufferSize);
    ~BufferDevice() override;
    void *GetBuffer(uint64_t bufferSize) override;

private:
    void Free();

private:
    void *buffer_ = nullptr;
    uint64_t bufferSize_ = 0;
};
} // namespace atb
#endif