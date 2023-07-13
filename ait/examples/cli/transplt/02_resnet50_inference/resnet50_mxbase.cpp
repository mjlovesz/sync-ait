#include <opencv2/opencv.hpp>
#include "MxBase/MxBase.h"


int main(int argc, char **argv)
{
    APP_ERROR ret = MxBase::MxInit();
    if (ret != APP_ERR_OK) return ret;

    // cv::dnn::Net net = cv::dnn::readNet("../assets/resnet50.onnx");
	// net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	// net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    uint32_t device_id = 0;
    std::string modelPath = "resnet50.om";
    MxBase::Model net(modelPath, device_id);

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat image = cv::imread("test.png", 1);
    if (image.empty()) return 0;

    cv::Mat resized_image, blob;
    cv::resize(image, resized_image, cv::Size(224, 224));
    cv::dnn::blobFromImage(resized_image, blob, 1.0, cv::Size(224, 224), cv::Scalar(), true, false);
    blob = (blob - cv::Scalar(123.675, 116.28, 103.53)) / cv::Scalar(58.395, 57.12, 57.375);

	auto end = std::chrono::high_resolution_clock::now();
    float time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Image process time duration: " << time_duration << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    /*
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    */

    const std::vector<uint32_t> shape = {1, 3, 224, 224};
    MxBase::Tensor tensor = MxBase::Tensor((void *)blob.data, shape, MxBase::TensorDType::FLOAT32, device_id);
    std::vector<MxBase::Tensor> mx_inputs = {tensor};
    std::vector<MxBase::Tensor> outputs = net.Infer(mx_inputs);
    outputs[0].ToHost();

	end = std::chrono::high_resolution_clock::now();
    time_duration= std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Model inference time duration: " << time_duration << "ms" << std::endl;

    int argmax = 0;
    float max_score = 0;
    float *data = (float *)outputs[0].GetData();
    for (int ii = 0; ii < 1000; ii++) {
        if (data[ii] > max_score) {
            max_score = data[ii];
            argmax = ii;
        }
    }
    std::cout << "index: " << argmax << std::endl;
    std::cout << "score: " << max_score << std::endl;
    
    return 0;
}
