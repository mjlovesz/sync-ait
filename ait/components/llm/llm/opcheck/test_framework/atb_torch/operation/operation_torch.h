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
#ifndef OPERATION_TORCH_H
#define OPERATION_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "atb/operation.h"
#include "hosttensor_binders/hosttensor_binder.h"

class OperationTorch : public torch::CustomClassHolder {
public:
    explicit OperationTorch(std::string opName);
    ~OperationTorch();
    void SetName(std::string name);
    std::string SetParam(std::string param);
    void SetVaraintPackParam(std::string varaintPackParam);
    std::string InferShape(std::vector<torch::Tensor> atInTensors);
    std::string Setup(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors);
    std::string ExecuteSync(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
        int64_t workspaceSize);
    std::vector<torch::Tensor> ExecuteWithParam(std::vector<torch::Tensor> atInTensors, std::string varaintPackParam);
    void ExecuteOutWithParam(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                             std::string varaintPackParam);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors);
    void ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors);
    c10::intrusive_ptr<OperationTorch> clone() const { return c10::make_intrusive<OperationTorch>(opName_); }

private:
    void CreateAtOutTensors(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    void ExecuteOutImpl(std::vector<torch::Tensor> &inTensors, std::vector<torch::Tensor> &outTensors);
    void BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    std::string GetSaveTensorDir();
    atb::Status InferShapeOutTensorDesc(std::vector<torch::Tensor> &atInTensors, 
                                        atb::Svector<atb::TensorDesc> &outTensorDescs);
    bool IsCopyStreamValid();

private:
    std::string opName_;
    uint64_t opId_ = 0;
    std::string nodeId_ = "0";
    std::string name_;
    std::string param_;
    std::unique_ptr<atb::Operation> operation_;
    std::unique_ptr<HostTensorBinder> hostTensorBinder_;
    uint64_t executeCount_ = 0;
    atb::VariantPack variantPack_;
};

#endif