#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

void resize_opencv(std::string imgFile) {
    cv::Mat img = cv::imread(imgFile, -1);
    cv::Size newSize = cv::Size(img.cols/2, img.rows/2);
    cv::Mat resizedImg(newSize, CV_8UC3);
    cv::resize(img, resizedImg, newSize, 0.0, 0.0, cv::INTER_LINEAR);
    cv::imwrite("./opencv_resize_output.jpg", resizedImg);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage ./resize_opencv xxx.jpg" << std::endl;
        return 1;
    }
    resize_opencv(argv[1]);
    cv::dnn dnn('placeholder');
    return 0;
}