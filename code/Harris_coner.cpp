#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// 정규화 함수
void normalizeMat(cv::Mat& src) {
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);

    src = (src - minVal) / (maxVal - minVal);
}

// 가우시안 커널 생성 함수
cv::Mat createGaussianKernel(int ksize, double sigma) {
    cv::Mat kernel(ksize, ksize, CV_32F);
    int k = ksize / 2;
    double sum = 0.0;

    for (int y = -k; y <= k; y++) {
        for (int x = -k; x <= k; x++) {
            kernel.at<float>(y + k, x + k) = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel.at<float>(y + k, x + k);
        }
    }

    // 커널 정규화
    for (int y = 0; y < ksize; y++) {
        for (int x = 0; x < ksize; x++) {
            kernel.at<float>(y, x) /= sum;
        }
    }

    return kernel;
}

// 가우시안 블러 적용 함수
void applyGaussianBlur(const cv::Mat& src, cv::Mat& dst, int ksize, double sigma) {
    cv::Mat kernel = createGaussianKernel(ksize, sigma);
    dst = cv::Mat::zeros(src.size(), CV_32F);
    int k = ksize / 2;

    // src 이미지를 float 타입으로 변환
    cv::Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    for (int y = k; y < src.rows - k; y++) {
        for (int x = k; x < src.cols - k; x++) {
            float sum = 0.0;
            for (int dy = -k; dy <= k; dy++) {
                for (int dx = -k; dx <= k; dx++) {
                    sum += srcFloat.at<float>(y + dy, x + dx) * kernel.at<float>(dy + k, dx + k);
                }
            }
            dst.at<float>(y, x) = sum;
        }
    }
}

// 소벨 연산을 이용해 첫 번째 및 두 번째 미분을 계산하는 함수
void applySobelDerivatives(const cv::Mat& gray, cv::Mat& grad_x, cv::Mat& grad_y, cv::Mat& dxx, cv::Mat& dyy, cv::Mat& dxy) {
    // 첫 번째 미분 소벨 커널
    cv::Mat sobel_x = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat sobel_y = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    // 두 번째 미분 소벨 커널
    cv::Mat sobel_xx = (cv::Mat_<float>(3, 3) << 1, -2, 1, 2, -4, 2, 1, -2, 1);
    cv::Mat sobel_yy = (cv::Mat_<float>(3, 3) << 1, 2, 1, -2, -4, -2, 1, 2, 1);
    cv::Mat sobel_xy = (cv::Mat_<float>(3, 3) << -1, 0, 1, 0, 0, 0, 1, 0, -1);

    // 출력 이미지 초기화
    grad_x = cv::Mat::zeros(gray.size(), CV_32F);
    grad_y = cv::Mat::zeros(gray.size(), CV_32F);
    dxx = cv::Mat::zeros(gray.size(), CV_32F);
    dyy = cv::Mat::zeros(gray.size(), CV_32F);
    dxy = cv::Mat::zeros(gray.size(), CV_32F);

    // 경계를 제외한 이미지 부분에 대해 소벨 연산 적용
    for (int y = 1; y < gray.rows - 1; y++) {
        for (int x = 1; x < gray.cols - 1; x++) {
            float gx = 0.0;
            float gy = 0.0;
            float gxx = 0.0;
            float gyy = 0.0;
            float gxy = 0.0;

            // 3x3 이웃에 대해 커널 적용
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    uchar pixel_value = gray.at<uchar>(y + ky, x + kx);
                    gx += pixel_value * sobel_x.at<float>(ky + 1, kx + 1);
                    gy += pixel_value * sobel_y.at<float>(ky + 1, kx + 1);
                    gxx += pixel_value * sobel_xx.at<float>(ky + 1, kx + 1);
                    gyy += pixel_value * sobel_yy.at<float>(ky + 1, kx + 1);
                    gxy += pixel_value * sobel_xy.at<float>(ky + 1, kx + 1);
                }
            }

            grad_x.at<float>(y, x) = gx;
            grad_y.at<float>(y, x) = gy;
            dxx.at<float>(y, x) = gxx;
            dyy.at<float>(y, x) = gyy;
            dxy.at<float>(y, x) = gxy;
        }
    }
}

void calculateHarrisResponse(const cv::Mat& dxx, const cv::Mat& dyy, const cv::Mat& dxy, cv::Mat& harrisResponse, double k) {
    harrisResponse = cv::Mat::zeros(dxx.size(), CV_32F);

    for (int y = 0; y < dxx.rows; y++) {
        for (int x = 0; x < dxx.cols; x++) {
            float dx2 = dxx.at<float>(y, x);
            float dy2 = dyy.at<float>(y, x);
            float dxy2 = dxy.at<float>(y, x);

            float detM = dx2 * dy2 - dxy2 * dxy2;
            if (detM < 0)
            {
                detM = 0;
            }
            else if (detM > 255)
            {
                detM = 255;
            }
            float traceM = dx2 + dy2;
            if (traceM < 0)
            {
                traceM = 0;
            }
            else if (traceM > 255)
            {
                traceM = 255;
            }
            float a = k * traceM * traceM;
            if (a < 0)
            {
                a = 0;
            }
            else if (a > 255)
            {
                a = 255;
            }
            float b = detM - a;
            if (b < 0)
            {
                b = 0;
            }
            else if (b > 255)
            {
                b = 255;
            }
            harrisResponse.at<float>(y, x) = b;
        }
    }
}

//비최대 억제
void nonMaximumSuppression(const cv::Mat& harrisResponse, cv::Mat& dst, int windowSize) {
    dst = cv::Mat::zeros(harrisResponse.size(), CV_32F);
    int border = windowSize / 2;

    for (int y = border; y < harrisResponse.rows - border; y++) {
        for (int x = border; x < harrisResponse.cols - border; x++) {
            float currentValue = harrisResponse.at<float>(y, x);
            bool isLocalMax = true;

            for (int dy = -border; dy <= border; dy++) {
                for (int dx = -border; dx <= border; dx++) {
                    if (dy == 0 && dx == 0) continue;

                    float neighborValue = harrisResponse.at<float>(y + dy, x + dx);
                    if (currentValue <= neighborValue) {
                        isLocalMax = false;
                        break;
                    }
                }
                if (!isLocalMax) break;
            }

            if (isLocalMax) {
                dst.at<float>(y, x) = currentValue;
            }
        }
    }
}

// 이미지 저장 함수
void saveDerivativeImages(const std::string& inputPath, const std::string& outputPath, const std::string& outputPathDx, const std::string& outputPathDy, const std::string& outputPathDxx, const std::string& outputPathDyy, const std::string& outputPathDxy, const std::string& outputPathGDxx, const std::string& outputPathGDyy, const std::string& outputPathGDxy) {
    // 이미지 로드
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image could not be loaded." << std::endl;
        return;
    }
    // 결과 저장
    cv::imwrite(outputPath, image);

    cv::Mat grad_x, grad_y, dxx, dyy, dxy;

    // 소벨 연산을 이용해 미분 계산
    applySobelDerivatives(image, grad_x, grad_y, dxx, dyy, dxy);

    // 가우시안 블러 적용 (직접 구현한 함수 사용)
    cv::Mat gdxx, gdyy, gdxy;
    applyGaussianBlur(dxx, gdxx, 5, 1);
    applyGaussianBlur(dyy, gdyy, 5, 1);
    applyGaussianBlur(dxy, gdxy, 5, 1);

    // 결과 저장
    cv::imwrite(outputPathDx, grad_x);
    cv::imwrite(outputPathDy, grad_y);
    cv::imwrite(outputPathDxx, dxx);
    cv::imwrite(outputPathDyy, dyy);
    cv::imwrite(outputPathDxy, dxy);
    cv::imwrite(outputPathGDxx, gdxx);
    cv::imwrite(outputPathGDyy, gdyy);
    cv::imwrite(outputPathGDxy, gdxy);

    // dxx, dyy, dxy 정규화
    normalizeMat(gdxx);
    normalizeMat(gdyy);
    normalizeMat(gdxy);

    // Harris 코너 응답 함수 계산
    cv::Mat harrisResponse;
    calculateHarrisResponse(gdxx, gdyy, gdxy, harrisResponse, 0.04);

    // 비최대 억제 적용
    cv::Mat nmsResponse;
    nonMaximumSuppression(harrisResponse, nmsResponse, 5);

    // 큰 값들만 255로 설정 (예: 최대값의 50% 이상)
    double maxVal;
    cv::minMaxLoc(nmsResponse, nullptr, &maxVal);
    double threshold = maxVal * 0.5;
    nmsResponse = (nmsResponse > threshold) * 255;
    nmsResponse.convertTo(nmsResponse, CV_8U);

    // 비최대 억제 적용 결과 저장
    cv::imwrite("nms_response.jpg", nmsResponse);
}

int main() {
    std::string inputPath = "building.jpg"; // 입력 이미지 경로
    std::string outputPath = "gray.jpg"; // dx 이미지 출력 경로
    std::string outputPathDx = "gradient_dx.jpg"; // dx 이미지 출력 경로
    std::string outputPathDy = "gradient_dy.jpg"; // dy 이미지 출력 경로
    std::string outputPathDxx = "gradient_dxx.jpg"; // dxx 이미지 출력 경로
    std::string outputPathDyy = "gradient_dyy.jpg"; // dyy 이미지 출력 경로
    std::string outputPathDxy = "gradient_dxy.jpg"; // dxy 이미지 출력 경로
    std::string outputPathGDxx = "gaussian_dxx.jpg"; // 가우시안 적용된 dxx 이미지 출력 경로
    std::string outputPathGDyy = "gaussian_dyy.jpg"; // 가우시안 적용된 dyy 이미지 출력 경로
    std::string outputPathGDxy = "gaussian_dxy.jpg"; // 가우시안 적용된 dxy 이미지 출력 경로

    saveDerivativeImages(inputPath, outputPath, outputPathDx, outputPathDy, outputPathDxx, outputPathDyy, outputPathDxy, outputPathGDxx, outputPathGDyy, outputPathGDxy);

    return 0;
}