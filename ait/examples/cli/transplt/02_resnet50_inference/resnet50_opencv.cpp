#include <opencv2/opencv.hpp>


int main(int argc, char **argv)
{
    cv::dnn::Net net = cv::dnn::readNet("resnet50.onnx");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

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
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

	end = std::chrono::high_resolution_clock::now();
    time_duration= std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Model inference time duration: " << time_duration << "ms" << std::endl;

    int argmax = 0;
    float max_score = 0;
    float *data = (float *)outputs[0].data;
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
