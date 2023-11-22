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
#include <opencv2/opencv.hpp>
#include "MxBase/MxBase.h"



int main(int argc, char **argv) {
    MxBase::MxInit();
    uint32_t device_id = 0;
    std::string modelPath = "resnet50_aipp_async.om";
    MxBase::Model net(modelPath, device_id);

    // 创建stream
    MxBase::AscendStream stream(device_id);
    stream.CreateAscendStream();

    auto start = std::chrono::high_resolution_clock::now();
    std::string img_file = "test.png";
    MxBase::ImageProcessor processor;
    MxBase::Image decoded_image;
    processor.Decode(img_file, decoded_image, MxBase::ImageFormat::RGB_888);
    MxBase::Image resized_image;
    processor.Resize(decoded_image, MxBase::Size(224, 224), resized_image, MxBase::Interpolation::HUAWEI_HIGH_ORDER_FILTER, stream);
    stream.Synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    float time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Image process time duration: " << time_duration << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    MxBase::Tensor tensor = resized_image.ConvertToTensor();
    tensor.ToDevice(device_id);
    std::vector<MxBase::Tensor> mx_inputs = {tensor};
    // 获取输出tensor的shape
    uint32_t index = 0;
    const std::vector<uint32_t> shape = net.GetOutputTensorShape(index);
    // 获取输出tensor的数量
    uint32_t num_outputs = net.GetOutputTensorNum();
    // 创建输出tensor的容器
    std::vector<MxBase::Tensor> outputs;
    outputs.resize(num_outputs);
    // 创建容器中tensor并分配内存
    for (size_t i = 0; i < num_outputs; ++i) {
        outputs[i] = MxBase::Tensor(shape, MxBase::TensorDType::FLOAT32, device_id);
        MxBase::Tensor::TensorMalloc(outputs[i]);}
    // 模型推理
    APP_ERROR ret = net.Infer(mx_inputs, outputs, stream);
    // 同步stream
    stream.Synchronize();
    outputs[0].ToHost();

    end = std::chrono::high_resolution_clock::now();
    time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Model inference time duration: " << time_duration << "ms" << std::endl;

    int total_classes = 1000;
    int argmax = 0;
    float max_score = 0;
    float *data = (float *)outputs[0].GetData();
    for (int ii = 0; ii < total_classes; ii++) {
        if (data[ii] > max_score) {
            max_score = data[ii];
            argmax = ii;
        }
    }
    std::cout << "index: " << argmax << std::endl;
    std::cout << "score: " << max_score << std::endl;

    return 0;
    // 销毁stram去初始化
    stream.DestroyAscendStream();
    MxBase::MxDeInit();
}