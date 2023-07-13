#include <opencv2/opencv.hpp>
#include "MxBase/MxBase.h"


int main(int argc, char **argv)
{
    APP_ERROR ret = MxBase::MxInit();
    if (ret != APP_ERR_OK) return ret;

    uint32_t device_id = 0;
    std::string modelPath = "resnet50_aipp.om";
    MxBase::Model net(modelPath, device_id);

    auto start = std::chrono::high_resolution_clock::now();
    std::string img_file = "test.png";
    MxBase::ImageProcessor processor;
    MxBase::Image decoded_image;
    processor.Decode(img_file, decoded_image, MxBase::ImageFormat::RGB_888);
    // MxBase::Image resized_image;
    // processor.Resize(decoded_image, MxBase::Size(256, 256), resized_image);
	auto end = std::chrono::high_resolution_clock::now();
    float time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
    std::cout << "Image process time duration: " << time_duration << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    MxBase::Tensor tensor = decoded_image.ConvertToTensor();
    // std::cout << tensor.GetByteSize() << std::endl;
    // std::cout << tensor.GetShape()[0] << ", " << tensor.GetShape()[1] << ", " << tensor.GetShape()[2] << ", " << tensor.GetShape()[3] << std::endl;
    tensor.ToDevice(device_id);
    std::vector<MxBase::Tensor> mx_inputs = {tensor};
    std::vector<MxBase::Tensor> outputs = net.Infer(mx_inputs);
    outputs[0].ToHost();

	end = std::chrono::high_resolution_clock::now();
    time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
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
