#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;

// 3x3 가로 방향 소벨 필터를 적용하는 함수 (8비트 이미지용)
Mat applySobelHorizontalFilter8U(const Mat& image) {
    Mat paddedImage;
    copyMakeBorder(image, paddedImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

    Mat result(image.size(), CV_8U, Scalar(0));
    int width = image.cols;
    int height = image.rows;

    // 가로 방향 소벨 필터
    float horizontalFilter[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    for (int y = 1; y < height + 1; ++y) {
        for (int x = 1; x < width + 1; ++x) {
            int horizontalGradient = 0;

            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    horizontalGradient += paddedImage.at<uchar>(y + j, x + i) * horizontalFilter[j + 1][i + 1];
                }
            }

            // 클램핑 대신 절대값 적용
            float absHorizontalGradient = std::abs(horizontalGradient);
            result.at<uchar>(y - 1, x - 1) = static_cast<uchar>(absHorizontalGradient);
        }
    }

    return result;
}

// 3x3 세로 방향 소벨 필터를 적용하는 함수 (8비트 이미지용)
Mat applySobelVerticalFilter8U(const Mat& image) {
    Mat paddedImage;
    copyMakeBorder(image, paddedImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

    Mat result(image.size(), CV_8U, Scalar(0));
    int width = image.cols;
    int height = image.rows;

    // 세로 방향 소벨 필터
    float verticalFilter[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    for (int y = 1; y < height + 1; ++y) {
        for (int x = 1; x < width + 1; ++x) {
            int verticalGradient = 0;

            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    verticalGradient += paddedImage.at<uchar>(y + j, x + i) * verticalFilter[j + 1][i + 1];
                }
            }

            // 클램핑 대신 절대값 적용
            float absVerticalGradient = std::abs(verticalGradient);
            result.at<uchar>(y - 1, x - 1) = static_cast<uchar>(absVerticalGradient);
        }

    }

    return result;
}

// 픽셀 값들을 텍스트로 이미지에 그리는 함수
Mat drawPixelValues(const Mat& img) {
    const int scale = 20; // 텍스트를 그릴 때 각 픽셀에 할당할 크기
    Mat result = Mat::zeros(img.rows * scale, img.cols * scale, CV_8UC3);

    // 텍스트 속성
    int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 0.5;
    int thickness = 1;

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            int pixelValue = img.at<uchar>(y, x); // 8비트 이미지의 픽셀 값
            string text = to_string(pixelValue);

            // 텍스트의 크기를 구함
            int baseline = 0;
            Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

            // 텍스트를 그릴 위치를 계산
            Point textOrg((x * scale + (scale - textSize.width) / 2), (y * scale + (scale + textSize.height) / 2));

            // 이미지에 텍스트를 그림
            putText(result, text, textOrg, fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
        }
    }

    return result;
}

// 에지 강도를 계산하는 함수 (8비트 이미지용)
Mat calculateEdgeIntensity8U(const Mat& horizontalImage, const Mat& verticalImage) {
    Mat edgeIntensity(horizontalImage.size(), CV_8U, Scalar(0));

    for (int y = 0; y < horizontalImage.rows; ++y) {
        for (int x = 0; x < horizontalImage.cols; ++x) {
            int gx = horizontalImage.at<uchar>(y, x);
            int gy = verticalImage.at<uchar>(y, x);
            edgeIntensity.at<uchar>(y, x) = saturate_cast<uchar>(sqrt(gx * gx + gy * gy));
        }
    }

    return edgeIntensity;
}

// 그레디언트 방향 계산 함수
Mat calculateGradientDirection(const Mat& horizontalImage, const Mat& verticalImage) {
    Mat gradientDirection(horizontalImage.size(), CV_32F, Scalar(0));

    for (int y = 0; y < horizontalImage.rows; ++y) {
        for (int x = 0; x < horizontalImage.cols; ++x) {
            int gx = horizontalImage.at<uchar>(y, x);
            int gy = verticalImage.at<uchar>(y, x);
            gradientDirection.at<float>(y, x) = atan2(gy, gx);
        }
    }

    return gradientDirection;
}

// 비최대 억제 함수
Mat nonMaximumSuppression(const Mat& edgeIntensityImage, const Mat& gradientDirection) {
    Mat outputImage = edgeIntensityImage.clone();

    int width = edgeIntensityImage.cols;
    int height = edgeIntensityImage.rows;

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float angle = gradientDirection.at<float>(y, x);

            // 각도를 4방향 중 하나로 근사화
            int direction = (int(round(angle / (M_PI / 4))) % 4 + 4) % 4;

            uchar currentPixel = edgeIntensityImage.at<uchar>(y, x);
            uchar pixel1 = 0, pixel2 = 0;

            switch (direction) {
            case 0: // 수평 방향
                pixel1 = edgeIntensityImage.at<uchar>(y, x - 1);
                pixel2 = edgeIntensityImage.at<uchar>(y, x + 1);
                break;
            case 1: // 대각선 방향 (우상향)
                pixel1 = edgeIntensityImage.at<uchar>(y - 1, x + 1);
                pixel2 = edgeIntensityImage.at<uchar>(y + 1, x - 1);
                break;
            case 2: // 수직 방향
                pixel1 = edgeIntensityImage.at<uchar>(y - 1, x);
                pixel2 = edgeIntensityImage.at<uchar>(y + 1, x);
                break;
            case 3: // 대각선 방향 (좌상향)
                pixel1 = edgeIntensityImage.at<uchar>(y - 1, x - 1);
                pixel2 = edgeIntensityImage.at<uchar>(y + 1, x + 1);
                break;
            }

            // 비최대 억제 적용
            if (currentPixel < pixel1 || currentPixel < pixel2) {
                outputImage.at<uchar>(y, x) = 0;
            }
        }
    }

    return outputImage;
}

Mat applyHysteresisThresholding(const Mat& edgeImage, uchar lowThreshold, uchar highThreshold) {
    Mat result = Mat::zeros(edgeImage.size(), CV_8U);

    for (int y = 0; y < edgeImage.rows; ++y) {
        for (int x = 0; x < edgeImage.cols; ++x) {
            uchar pixelValue = edgeImage.at<uchar>(y, x);

            if (pixelValue >= highThreshold) {
                result.at<uchar>(y, x) = 255; // 강한 엣지
            }
            else if (pixelValue >= lowThreshold) {
                // 주변 픽셀 확인
                for (int j = -1; j <= 1; ++j) {
                    for (int i = -1; i <= 1; ++i) {
                        if (y + j >= 0 && y + j < edgeImage.rows && x + i >= 0 && x + i < edgeImage.cols) {
                            if (edgeImage.at<uchar>(y + j, x + i) >= highThreshold) {
                                result.at<uchar>(y, x) = 255; // 강한 엣지와 연결된 약한 엣지
                                break;
                            }
                        }
                    }
                    if (result.at<uchar>(y, x) == 255) {
                        break;
                    }
                }
            }
        }
    }

    return result;
}

// 원 중복 제거 함수
vector<Vec3f> nonMaximumSuppressionCircles(const vector<Vec3f>& circles, float distanceThreshold, float radiusThreshold) {
    vector<Vec3f> result;
    for (const auto& circle : circles) {
        bool keep = true;
        for (const auto& selectedCircle : result) {
            float dist = sqrt(pow(circle[0] - selectedCircle[0], 2) + pow(circle[1] - selectedCircle[1], 2));
            float radiusDiff = abs(circle[2] - selectedCircle[2]);

            if (dist < distanceThreshold && radiusDiff < radiusThreshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            result.push_back(circle);
        }
    }
    return result;
}

// 허프 원 변환 함수
vector<Vec3f> applyHoughCircleTransform(const Mat& edgeImage, int minRadius, int maxRadius, int threshold) {
    int rows = edgeImage.rows;
    int cols = edgeImage.cols;
    vector<Vec3f> circles;

    // 각 반지름에 대한 누적 배열
    for (int radius = minRadius; radius <= maxRadius; radius++) {
        Mat accumulator = Mat::zeros(rows, cols, CV_32S);

        // 에지 픽셀에 대해 허프 변환 수행
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (edgeImage.at<uchar>(y, x) > 0) { // 에지 픽셀 검사
                    for (int theta = 0; theta < 360; theta++) {
                        int a = round(x - radius * cos(theta * CV_PI / 180));
                        int b = round(y - radius * sin(theta * CV_PI / 180));
                        if (a >= 0 && a < cols && b >= 0 && b < rows) {
                            accumulator.at<int>(b, a)++;
                        }
                    }
                }
            }
        }

        // 임계값 이상의 누적값을 가진 원 찾기
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (accumulator.at<int>(y, x) > threshold) {
                    circles.push_back(Vec3f(x, y, radius));
                }
            }
        }
    }

    // 비최대 억제 적용
    float distanceThreshold = 10.0f; // 중심점 간 최소 거리
    float radiusThreshold = 5.0f; // 반지름 차이 임계값
    return nonMaximumSuppressionCircles(circles, distanceThreshold, radiusThreshold);
}

int main() {
    // 이미지를 그레이스케일로 로드
    Mat image = imread("./canyy edge/hurfone/smarties2.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Image cannot be loaded." << endl;
        return -1;
    }
    Mat imagetext = drawPixelValues(image);
    imwrite("./canyy edge/hurfone/sobel_image_text.jpg", imagetext); // 저장

    // 가로 방향 소벨 필터를 적용하여 이미지 생성 (8비트 이미지용)
    Mat sobelHorizontalImage = applySobelHorizontalFilter8U(image);
    imwrite("./canyy edge/hurfone/sobel_horizontal.jpg", sobelHorizontalImage); // 저장

    // 세로 방향 소벨 필터를 적용하여 이미지 생성 (8비트 이미지용)
    Mat sobelVerticalImage = applySobelVerticalFilter8U(image);
    imwrite("./canyy edge/hurfone/sobel_vertical.jpg", sobelVerticalImage); // 저장

    // 텍스트화한 이미지 생성 및 저장
    Mat sobelHorizontalTextImage = drawPixelValues(sobelHorizontalImage);
    Mat sobelVerticalTextImage = drawPixelValues(sobelVerticalImage);
    imwrite("./canyy edge/hurfone/sobel_horizontal_text.jpg", sobelHorizontalTextImage);
    imwrite("./canyy edge/hurfone/sobel_vertical_text.jpg", sobelVerticalTextImage);

    // 에지 강도 계산 및 저장
    Mat edgeIntensityImage = calculateEdgeIntensity8U(sobelHorizontalImage, sobelVerticalImage);
    imwrite("./canyy edge/hurfone/edge_intensity.jpg", edgeIntensityImage); // 저장

    // 에지 강도 텍스트화한 이미지 생성 및 저장
    Mat edgeIntensityTextImage = drawPixelValues(edgeIntensityImage);
    imwrite("./canyy edge/hurfone/edge_intensity_text.jpg", edgeIntensityTextImage);


    // 그레디언트 방향 계산
    Mat gradientDirection = calculateGradientDirection(sobelHorizontalImage, sobelVerticalImage);

    // 비최대 억제 적용
    Mat nonMaxSuppressedImage = nonMaximumSuppression(edgeIntensityImage, gradientDirection);
    imwrite("./canyy edge/hurfone/non_max_suppressed.jpg", nonMaxSuppressedImage);

    // Hysteresis Thresholding 적용
    uchar lowThreshold = 50;   // 낮은 임계값 설정
    uchar highThreshold = 100; // 높은 임계값 설정
    Mat hysteresisImage = applyHysteresisThresholding(nonMaxSuppressedImage, lowThreshold, highThreshold);
    imwrite("./canyy edge/hurfone/hysteresis_thresholded.jpg", hysteresisImage);

    // Hough 원 변환 적용
    int minRadius = 10;   // 최소 반지름 설정
    int maxRadius = 100;  // 최대 반지름 설정
    int threshold = 150;   // 임계값 설정
    vector<Vec3f> circles = applyHoughCircleTransform(hysteresisImage, minRadius, maxRadius, threshold);

    // 원 표시
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        circle(hysteresisImage, Point(c[0], c[1]), c[2], Scalar(255, 255, 255), LINE_AA);
    }

    // 결과 이미지 저장
    imwrite("./canyy edge/hurfone/detected_circles.jpg", hysteresisImage);

    return 0;
}
