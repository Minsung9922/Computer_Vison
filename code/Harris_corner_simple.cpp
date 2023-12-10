#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
using namespace std;

// 소벨 필터를 이용한 그라디언트 계산 함수
void manualSobelGradient(const cv::Mat& src, cv::Mat& grad_x, cv::Mat& grad_y) {
    // 소벨 커널 정의
    cv::Mat sobel_x = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat sobel_y = (cv::Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    // 이미지에 패딩 적용
    cv::Mat padded_src;
    int pad = 1;
    cv::copyMakeBorder(src, padded_src, pad, pad, pad, pad, cv::BORDER_REPLICATE);

    // 그라디언트 매트릭스 초기화
    grad_x = cv::Mat::zeros(src.size(), CV_64F);
    grad_y = cv::Mat::zeros(src.size(), CV_64F);

    // 컨볼루션 수행
    for (int y = pad; y < padded_src.rows - pad; ++y) {
        for (int x = pad; x < padded_src.cols - pad; ++x) {
            double sum_x = 0.0, sum_y = 0.0;
            for (int ky = -pad; ky <= pad; ++ky) {
                for (int kx = -pad; kx <= pad; ++kx) {
                    sum_x += padded_src.at<uchar>(y + ky, x + kx) * sobel_x.at<double>(ky + pad, kx + pad);
                    sum_y += padded_src.at<uchar>(y + ky, x + kx) * sobel_y.at<double>(ky + pad, kx + pad);
                }
            }
            grad_x.at<double>(y - pad, x - pad) = sum_x;
            grad_y.at<double>(y - pad, x - pad) = sum_y;
        }
    }
}


// 그라디언트 제곱 및 곱셈 값을 계산하는 함수
void calculateGradientProducts(const cv::Mat& grad_x, const cv::Mat& grad_y,
    cv::Mat& grad_xx, cv::Mat& grad_yy, cv::Mat& grad_xy) {
    grad_xx = cv::Mat::zeros(grad_x.size(), CV_64F);
    grad_yy = cv::Mat::zeros(grad_y.size(), CV_64F);
    grad_xy = cv::Mat::zeros(grad_x.size(), CV_64F);

    for (int y = 0; y < grad_x.rows; y++) {
        for (int x = 0; x < grad_x.cols; x++) {
            grad_xx.at<double>(y, x) = grad_x.at<double>(y, x) * grad_x.at<double>(y, x);
            grad_yy.at<double>(y, x) = grad_y.at<double>(y, x) * grad_y.at<double>(y, x);
            grad_xy.at<double>(y, x) = grad_x.at<double>(y, x) * grad_y.at<double>(y, x);
        }
    }
}

// 픽셀 값들을 텍스트로 이미지에 그리는 함수
cv::Mat drawPixelValues(const cv::Mat& img) {
    const int scale = 20; // 텍스트를 그릴 때 각 픽셀에 할당할 크기
    cv::Mat result = cv::Mat::zeros(img.rows * scale, img.cols * scale, CV_8UC3);

    int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 0.5;
    int thickness = 1;

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            std::string text;
            if (img.type() == CV_64F) {
                // 이미지가 double 타입일 경우
                text = std::to_string(static_cast<int>(img.at<double>(y, x)));
            }
            else if (img.type() == CV_8U) {
                // 이미지가 uchar 타입일 경우
                text = std::to_string(img.at<uchar>(y, x));
            }

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
            cv::Point textOrg((x * scale + (scale - textSize.width) / 2), (y * scale + (scale + textSize.height) / 2));

            cv::putText(result, text, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }
    }

    return result;
}


int main() {
    // Step 1: 10x10 검은색 이미지 생성
    cv::Mat image = cv::Mat::zeros(9, 9, CV_8UC1);

    // Step 2: 가운데 5x5 흰색 사각형 추가 후 이미지 저장
    cv::rectangle(image, cv::Point(2, 2), cv::Point(6, 6), cv::Scalar(255), -1);
    cv::imwrite("step2_white_square.png", image);

    // Step 2의 이미지에 픽셀 값들을 텍스트로 그리고 저장합니다.
    cv::Mat image_with_values = drawPixelValues(image);
    cv::imwrite("step2_white_square_values.png", image_with_values);

    // Step 3: 그라디언트 계산
    cv::Mat grad_x, grad_y;
    manualSobelGradient(image, grad_x, grad_y);  // 수정된 함수 사용

    // 이미지 저장
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::imwrite("step3_grad_x.png", abs_grad_x);
    cv::imwrite("step3_grad_y.png", abs_grad_y);

    // Step 4: 그라디언트 제곱 및 곱셈 이미지 계산
    cv::Mat grad_xx, grad_yy, grad_xy;
    calculateGradientProducts(grad_x, grad_y, grad_xx, grad_yy, grad_xy);

    // 이미지 저장
    cv::Mat abs_grad_xx, abs_grad_yy, abs_grad_xy;
    cv::convertScaleAbs(grad_xx, abs_grad_xx);
    cv::convertScaleAbs(grad_yy, abs_grad_yy);
    cv::convertScaleAbs(grad_xy, abs_grad_xy);
    cv::imwrite("step4_grad_xx.png", abs_grad_xx);
    cv::imwrite("step4_grad_yy.png", abs_grad_yy);
    cv::imwrite("step4_grad_xy.png", abs_grad_xy);

    // Step 5: 해리스 코너 스코어 계산 및 절대값 적용, 반올림
    double k = 0.04;
    cv::Mat harris_response = cv::Mat::zeros(grad_xx.size(), CV_64F);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            double gxx = grad_xx.at<double>(y, x);
            double gyy = grad_yy.at<double>(y, x);
            double gxy = grad_xy.at<double>(y, x);
            double det = (gxx * gyy) - (gxy * gxy);
            double trace = gxx + gyy;
            double response = det - k * trace * trace;

            // 음수값 처리 및 반올림
            response = std::abs(response);
            response = std::round(response);

            cout << "(" << x << ", " << y << ") → C : " << (int)response << endl;

            harris_response.at<double>(y, x) = response;
        }
    }



    // 이미지 저장
    cv::Mat norm_harris_response;
    cv::normalize(harris_response, norm_harris_response, 0, 255, cv::NORM_MINMAX, CV_32FC1);
    cv::Mat scaled_harris_response;
    cv::convertScaleAbs(norm_harris_response, scaled_harris_response);
    cv::imwrite("step5_harris_response.png", scaled_harris_response);

    // Step 6: 비최대 억제 수행
    cv::Mat suppressed = cv::Mat::zeros(harris_response.size(), CV_64F);
    for (int y = 1; y < harris_response.rows - 1; ++y) {
        for (int x = 1; x < harris_response.cols - 1; ++x) {
            double value = harris_response.at<double>(y, x);
            if (value > harris_response.at<double>(y - 1, x - 1) &&
                value > harris_response.at<double>(y - 1, x) &&
                value > harris_response.at<double>(y - 1, x + 1) &&
                value > harris_response.at<double>(y, x - 1) &&
                value > harris_response.at<double>(y, x + 1) &&
                value > harris_response.at<double>(y + 1, x - 1) &&
                value > harris_response.at<double>(y + 1, x) &&
                value > harris_response.at<double>(y + 1, x + 1)) {
                suppressed.at<double>(y, x) = 255;
            }
            else {
                suppressed.at<double>(y, x) = 0;
            }
        }
    }

    // 이미지 저장
    cv::Mat suppressed_u8;
    suppressed.convertTo(suppressed_u8, CV_8U);
    cv::imwrite("step6_suppressed.png", suppressed_u8);

    // 생성된 모든 이미지에 대해 픽셀 값들을 텍스트로 그리고 저장합니다.
    std::vector<cv::Mat> images_to_draw = {
        abs_grad_x, abs_grad_y, abs_grad_xx, abs_grad_yy, abs_grad_xy, scaled_harris_response, suppressed_u8
    };
    std::vector<std::string> image_names = {
        "step3_grad_x_values.png", "step3_grad_y_values.png", "step4_grad_xx_values.png",
        "step4_grad_yy_values.png", "step4_grad_xy_values.png", "step5_harris_response_values.png", "step6_suppressed_values.png"
    };

    for (size_t i = 0; i < images_to_draw.size(); ++i) {
        cv::Mat drawn_image = drawPixelValues(images_to_draw[i]);
        cv::imwrite(image_names[i], drawn_image);
    }

    return 0;
}