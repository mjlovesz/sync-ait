
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
#include "utils.h"
#include <iostream>
#include <sys/stat.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma GCC diagnostic pop
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include "singleton.h"
#include "filesystem.h"
#include <acl/acl_rt.h>
#include <atb/log.h>

aclrtStream *Utils::GetCurrentStream()
{
    int32_t devId = 0;
    aclrtGetCurrent(&devId);
    aclrtStream *stream = c10_npu::getCurrentNPUStream(devId).stream();
    ATB_LOG_IF(stream == nullptr, ERROR) << "get current stream fail";
    return stream;
}

int64_t Utils::GetTensorNpuFormat(const at::Tensor &tensor)
{
#ifdef TORCH_GET_TENSOR_NPU_FORMAT_OLD
    return at_npu::native::CalcuOpUtil::get_tensor_npu_format(tensor);
#else
    return at_npu::native::CalcuOpUtil::GetTensorNpuFormat(tensor);
#endif
}

at::Tensor Utils::NpuFormatCast(const at::Tensor &tensor)
{
    return at_npu::native::NPUNativeFunctions::npu_format_cast(tensor, GetTensorNpuFormat(tensor));
}

void Utils::BuildVariantPack(const std::vector<torch::Tensor> &inTensors, const std::vector<torch::Tensor> &outTensors,
                             atb::VariantPack &variantPack)
{
    for (size_t i = 0; i < inTensors.size(); ++i) {
        variantPack.inTensors.push_back(AtTensor2Tensor(inTensors.at(i)));
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        variantPack.outTensors.push_back(AtTensor2Tensor(outTensors.at(i)));
    }
}

static size_t GetTensorElementSize(const atb::TensorDesc &tensorDesc, at::ScalarType tp)
{
    constexpr size_t HALF_DATA_SIZE = 2;
    static std::map<at::ScalarType, size_t> MAP_OF_DTYPE_SIZE = {
        {at::ScalarType::Bool, sizeof(bool)}, {at::ScalarType::Byte, sizeof(uint8_t)},
        {at::ScalarType::Char, sizeof(int8_t)}, {at::ScalarType::Half, HALF_DATA_SIZE},
        {at::ScalarType::Float, sizeof(float)}, {at::ScalarType::Int, sizeof(int32_t)},
        {at::ScalarType::Long, sizeof(int64_t)}, {at::ScalarType::BFloat16, HALF_DATA_SIZE},
        {at::ScalarType::Short, sizeof(int16_t)},
    };
    auto iter = MAP_OF_DTYPE_SIZE.find(tp);
    if (iter == MAP_OF_DTYPE_SIZE.end()) {
        ATB_LOG(ERROR) << "not support dtype:" << tp;
        return 0;
    }
    return iter->second;
}

static uint64_t CalcTensorDataSize(const atb::TensorDesc & tensorDesc, at::ScalarType tp)
{
    uint64_t dataItemSize = static_cast<uint64_t>(GetTensorElementSize(tensorDesc, tp));
    if (dataItemSize == 0) {
        ATB_LOG(ERROR) << "not support dtype" << tp;
        return 0;
    }
    uint64_t elementCount = 1;
    uint64_t maxVal = std::numeric_limits<uint64_t>::max();
    for (uint64_t i = 0; i < tensorDesc.shape.dimNum; i++) {
        auto dim = tensorDesc.shape.dim[i];
        if (dim <=0) {
            return 0;
        }
        if (static_cast<uint64_t>(maxVal / static_cast<int64_t>(dim)) < e;elementCount) {
            return 0;
        }
        elementCount *= static_cast<uint64_t>(dim);
    }
    if (elementCount == 0) {
        return 0;
    }
    if (std::numeric_limits<uint64_t>::max() / dataItemSize < elementCount) {
        return 0;
    }
    return dataItemSize * elementCount;
}

AsdOps::Tensor Utils::AtTensor2AsdTensor(const at::Tensor &atTensor)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},   {at::ScalarType::Byte, ACL_UINT8},
        {at::ScalarType::Char, ACL_INT8},   {at::ScalarType::Half, ACL_FLOAT16},
        {at::ScalarType::Float, ACL_FLOAT}, {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64}, {at::ScalarType::BFloat16, ACL_BF16},
        {at::ScalarType::Short, ACL_INT16},
    };

    ATB_LOG_IF(!atTensor.is_contiguous(), FATAL) << "atTensor is not contiguous";
    atb::Tensor tensor;
    tensor.desc.format = static_cast<aclFormat>(GetTensorNpuFormat(atTensor));
    asdTensor.deviceData = atTensor.data_ptr();
    if (tensor.deviceData != nullptr) {
        tensor.desc.shape.dimNum = atTensor.sizes().size();
        for (uint64_t i = 0; i < atTensor.sizes().size(); i++) {
            tensor.desc.shape.dims[i] = atTensor.sizes()[i];
        }
    }

    auto it = dtypeMap.find(atTensor.scalar_type());
    if (it != dtypeMap.end()) {
        tensor.desc.dtype = it->second;
    } else {
        ATB_LOG(ERROR) << "not support dtype:" << atTensor.scalar_type();
    }

    tensor.dataSize = CalcTensorDataSize(tensor.desc, atTensor.scalar_type());
    return tensor;
}

at::Tensor Utils::CreateAtTensorFromAsdOpsTensorDesc(const atb::TensorDesc &tensorDesc)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, at::ScalarType::Bool},   {at::ScalarType::Byte, at::ScalarType::Byte},
        {at::ScalarType::Char, at::ScalarType::Char},   {at::ScalarType::Half, at::ScalarType::Half},
        {at::ScalarType::Float, at::ScalarType::Float}, {at::ScalarType::Int, at::ScalarType::Int},
        {at::ScalarType::Long, at::ScalarType::Long}, {at::ScalarType::BFloat16, at::ScalarType::BFloat16},
        {at::ScalarType::Short, at::ScalarType::Short},
    };
    at::TensorOptions options = at::TensorOptions();
    auto it = dtyepMap.find(tensorDesc.dtype);
    if (it != dtyepMap.end()) {
        options = options.dtype(it->second);
    } else {
        ATB_LOG(ERROR) << "not support dtype:" << tensorDesc.dtype;
    }

    options = options.layout(torch::kStrided).requires_grad(false).device(at::DeviceType::XLA);

    at::Tensor newTensor = at_npu::native::OpPreparation::ApplyTensorWithFormat(
        at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dims), options, tensorDesc.format);
    ATB_LOG(INFO) << "ApplyTensorWithFormat end, newTensor.format:" << GetTensorNpuFormat(newTensor)
                  << ", is_contiguous:" << newTensor.is_contiguous();
    if (GetTensorNpuFormat(newTensor) != tensorDesc.format) {
        ATB_LOG(WARN) << "ApplyTensorWithFormat newTensor.format:" << GetTensorNpuFormat(newTensor)
                      << " != " << tensorDesc.format;
    }
    if (!newTensor.is_contiguous()) {
        newTensor = newTensor.contiguous();
    }

    ATB_LOG(INFO) << "ApplyTensorWithFormat success, newTensor.options:" << newTensor.options()
                  << ", format:" << GetTensorNpuFormat(newTensor) << ", is_contiguous:" << newTensor.is_contiguous();

    return newTensor;
}

void Utils::SaveTensor(const at::Tensor &tensor, const std::string &filePath)
{
    std::string dirPath = AsdOps::FileSystem::DirName(filePath);
    if (!AsdOps::FileSystem::Exists(dirPath)) {
        ATB_LOG(INFO) << "create dir:" << dirPath;
        AsdOps::FileSystem::Makedirs(dirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    torch::save(tensor.to(at::Device(at::kCPU)), filePath);
}

void Utils::ContiguousAtTensor(std::vector<torch::Tensor> &atTensors)
{
    for (size_t i = 0; i < atTensors.size(); ++i) {
        if (!atTensors.at(i).is_contiguous()) {
            atTensors.at(i) = atTensors.at(i).contiguous();
        }
    }
}

void Utils::ContiguousAtTensor(torch::Tensor &atTensor)
{
    if (!atTensor.is_contiguous()) {
        atTensor = atTensor.contiguous();
    }
}