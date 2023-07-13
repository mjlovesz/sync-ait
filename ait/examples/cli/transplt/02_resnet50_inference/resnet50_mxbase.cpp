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
    APP_ERROR ret = MxBase::MxInit();
    if (ret != APP_ERR_OK) return ret;

    uint32_t device_id = 0;
    std::string modelPath = "resnet50.om";
    MxBase::Model net(modelPath, device_id);

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat image = cv::imread("test.png", 1);
    if (image.empty()) return 0;

    int input_height = 224;
    int input_width = 224;
    auto mean_channel = cv::Scalar(123.675, 116.28, 103.53);
    auto std_channel = cv::Scalar(58.395, 57.12, 57.375);
    cv::Mat resized_image, blob;
    cv::resize(image, resized_image, cv::Size(input_height, input_width));
    cv::dnn::blobFromImage(resized_image, blob, 1.0, cv::Size(input_height, input_width), cv::Scalar(), true, false);
    blob = (blob - mean_channel) / std_channel;

    auto end = std::chrono::high_resolution_clock::now();
    float time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Image process time duration: " << time_duration << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    const std::vector<uint32_t> shape = {1, 3, 224, 224};
    MxBase::Tensor tensor = MxBase::Tensor((void *)blob.data, shape, MxBase::TensorDType::FLOAT32, device_id);
    std::vector<MxBase::Tensor> mx_inputs = {tensor};
    std::vector<MxBase::Tensor> outputs = net.Infer(mx_inputs);
    outputs[0].ToHost();

    end = std::chrono::high_resolution_clock::now();
    time_duration= std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
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
}
