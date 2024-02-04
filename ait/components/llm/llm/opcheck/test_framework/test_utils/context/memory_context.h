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
#ifndef ATB_CONTEXT_CONTEXT_H
#define ATB_CONTEXT_CONTEXT_H
#include <cstdint>
#include <memory>
#include <vector>

namespace atb {
class BufferBase;

class MemoryContext {
public:
    MemoryContext();
    ~MemoryContext();
    void *GetHostTilingBuffer(uint64_t bufferSize);
    void *GetTilingBuffer(uint64_t bufferSize);
    void *GetWorkspaceBuffer(uint64_t bufferSize);
    void *GetIntermediateBuffer(uint64_t bufferSize);

private:
    uint64_t GetHostTilingBufferRing() const;
    uint64_t GetHostTilingBufferSize() const;

    uint64_t GetTilingBufferRing() const;
    uint64_t GetTilingBufferSize() const;

    uint64_t GetWorkspaceBufferRing() const;
    uint64_t GetWorkspaceBufferSize() const;

    uint64_t GetIntermediateBufferRing() const;
    uint64_t GetIntermediateBufferSize() const;

private:
    std::vector<std::unique_ptr<BufferBase>> hostTilingBuffers_;
    size_t hostTilingBufferOffset_ = 0;
    std::vector<std::unique_ptr<BufferBase>> tilingBuffers_;
    size_t tilingBufferOffset_ = 0;
    std::vector<std::unique_ptr<BufferBase>> workspaceBuffers_;
    size_t workspaceBufferOffset_ = 0;
    std::vector<std::unique_ptr<BufferBase>> intermediateBuffers_;
    size_t intermediateBufferOffset_ = 0;
};
} // namespace atb
#endif