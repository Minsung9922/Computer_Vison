#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

// 가우시안 커널을 계산하는 함수
void calculateGaussianKernel(vector<vector<double>>& kernel, int kernelSize, double sigma) {
    const int halfSize = kernelSize / 2;
    double sum = 0.0; // 커널의 합을 계산하기 위한 변수
    kernel.resize(kernelSize, vector<double>(kernelSize, 0));

    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            kernel[i + halfSize][j + halfSize] = exp(-(i * i + j * j) / (2 * sigma * sigma));
            sum += kernel[i + halfSize][j + halfSize];
        }
    }

    // 커널 정규화
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= sum;
        }
    }
}

// 이미지에 패딩을 추가하는 함수
Mat addPadding(const Mat& inputImage, int padSize) {
    Mat paddedImage(inputImage.rows + 2 * padSize, inputImage.cols + 2 * padSize, inputImage.type(), Scalar::all(0));

    // 입력 이미지를 새로운 패딩 이미지의 중앙에 복사
    inputImage.copyTo(paddedImage(Rect(padSize, padSize, inputImage.cols, inputImage.rows)));

    // 테두리 복사를 통해 패딩 추가
    for (int i = 0; i < padSize; ++i) {
        // 상단 패딩
        paddedImage.row(padSize).copyTo(paddedImage.row(i));
        // 하단 패딩
        paddedImage.row(paddedImage.rows - padSize - 1).copyTo(paddedImage.row(paddedImage.rows - i - 1));
        // 좌측 패딩
        paddedImage.col(padSize).copyTo(paddedImage.col(i));
        // 우측 패딩
        paddedImage.col(paddedImage.cols - padSize - 1).copyTo(paddedImage.col(paddedImage.cols - i - 1));
    }

    return paddedImage;
}

// 가우시안 필터를 이미지에 적용하는 함수
void applyGaussianFilter(const Mat& inputImage, Mat& outputImage, int kernelSize, double sigma) {
    const int halfSize = kernelSize / 2;
    vector<vector<double>> kernel;

    // 가우시안 커널 계산
    calculateGaussianKernel(kernel, kernelSize, sigma);

    // 입력 이미지에 패딩 추가
    Mat paddedImage = addPadding(inputImage, halfSize);

    // 결과 이미지 초기화
    outputImage = Mat::zeros(inputImage.size(), inputImage.type());

    // 패딩된 이미지에 가우시안 필터 적용
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            double filteredPixel = 0.0;

            for (int i = -halfSize; i <= halfSize; ++i) {
                for (int j = -halfSize; j <= halfSize; ++j) {
                    // 커널과 이미지의 상응하는 픽셀값을 곱하여 누적합
                    filteredPixel += kernel[i + halfSize][j + halfSize] * paddedImage.at<uchar>(y + i + halfSize, x + j + halfSize);
                }
            }

            outputImage.at<uchar>(y, x) = saturate_cast<uchar>(filteredPixel);
        }
    }
}

int main() {
    // 이미지를 그레이스케일로 불러오기
    Mat image = imread("apple.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "이미지를 불러올 수 없습니다." << endl;
        return -1;
    }

    // 필터링된 이미지를 저장할 객체
    Mat filteredImage;

    // 사용할 커널의 크기와 시그마 값을 정의
    const int kernelSize = 5;
    const double sigma = 1.0;

    // 가우시안 필터를 직접 적용하는 함수 호출
    applyGaussianFilter(image, filteredImage, kernelSize, sigma);

    // 필터링된 이미지를 저장
    imwrite("apple_filtered_sigma_2.jpg", filteredImage);

    cout << "가우시안 필터가 적용된 이미지를 저장했습니다." << endl;

    return 0;
}
