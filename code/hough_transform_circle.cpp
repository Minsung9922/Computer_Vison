#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;

// 3x3 ���� ���� �Һ� ���͸� �����ϴ� �Լ� (8��Ʈ �̹�����)
Mat applySobelHorizontalFilter8U(const Mat& image) {
    Mat paddedImage;
    copyMakeBorder(image, paddedImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

    Mat result(image.size(), CV_8U, Scalar(0));
    int width = image.cols;
    int height = image.rows;

    // ���� ���� �Һ� ����
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

            // Ŭ���� ��� ���밪 ����
            float absHorizontalGradient = std::abs(horizontalGradient);
            result.at<uchar>(y - 1, x - 1) = static_cast<uchar>(absHorizontalGradient);
        }
    }

    return result;
}

// 3x3 ���� ���� �Һ� ���͸� �����ϴ� �Լ� (8��Ʈ �̹�����)
Mat applySobelVerticalFilter8U(const Mat& image) {
    Mat paddedImage;
    copyMakeBorder(image, paddedImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

    Mat result(image.size(), CV_8U, Scalar(0));
    int width = image.cols;
    int height = image.rows;

    // ���� ���� �Һ� ����
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

            // Ŭ���� ��� ���밪 ����
            float absVerticalGradient = std::abs(verticalGradient);
            result.at<uchar>(y - 1, x - 1) = static_cast<uchar>(absVerticalGradient);
        }

    }

    return result;
}

// �ȼ� ������ �ؽ�Ʈ�� �̹����� �׸��� �Լ�
Mat drawPixelValues(const Mat& img) {
    const int scale = 20; // �ؽ�Ʈ�� �׸� �� �� �ȼ��� �Ҵ��� ũ��
    Mat result = Mat::zeros(img.rows * scale, img.cols * scale, CV_8UC3);

    // �ؽ�Ʈ �Ӽ�
    int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 0.5;
    int thickness = 1;

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            int pixelValue = img.at<uchar>(y, x); // 8��Ʈ �̹����� �ȼ� ��
            string text = to_string(pixelValue);

            // �ؽ�Ʈ�� ũ�⸦ ����
            int baseline = 0;
            Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

            // �ؽ�Ʈ�� �׸� ��ġ�� ���
            Point textOrg((x * scale + (scale - textSize.width) / 2), (y * scale + (scale + textSize.height) / 2));

            // �̹����� �ؽ�Ʈ�� �׸�
            putText(result, text, textOrg, fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
        }
    }

    return result;
}

// ���� ������ ����ϴ� �Լ� (8��Ʈ �̹�����)
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

// �׷����Ʈ ���� ��� �Լ�
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

// ���ִ� ���� �Լ�
Mat nonMaximumSuppression(const Mat& edgeIntensityImage, const Mat& gradientDirection) {
    Mat outputImage = edgeIntensityImage.clone();

    int width = edgeIntensityImage.cols;
    int height = edgeIntensityImage.rows;

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float angle = gradientDirection.at<float>(y, x);

            // ������ 4���� �� �ϳ��� �ٻ�ȭ
            int direction = (int(round(angle / (M_PI / 4))) % 4 + 4) % 4;

            uchar currentPixel = edgeIntensityImage.at<uchar>(y, x);
            uchar pixel1 = 0, pixel2 = 0;

            switch (direction) {
            case 0: // ���� ����
                pixel1 = edgeIntensityImage.at<uchar>(y, x - 1);
                pixel2 = edgeIntensityImage.at<uchar>(y, x + 1);
                break;
            case 1: // �밢�� ���� (�����)
                pixel1 = edgeIntensityImage.at<uchar>(y - 1, x + 1);
                pixel2 = edgeIntensityImage.at<uchar>(y + 1, x - 1);
                break;
            case 2: // ���� ����
                pixel1 = edgeIntensityImage.at<uchar>(y - 1, x);
                pixel2 = edgeIntensityImage.at<uchar>(y + 1, x);
                break;
            case 3: // �밢�� ���� (�»���)
                pixel1 = edgeIntensityImage.at<uchar>(y - 1, x - 1);
                pixel2 = edgeIntensityImage.at<uchar>(y + 1, x + 1);
                break;
            }

            // ���ִ� ���� ����
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
                result.at<uchar>(y, x) = 255; // ���� ����
            }
            else if (pixelValue >= lowThreshold) {
                // �ֺ� �ȼ� Ȯ��
                for (int j = -1; j <= 1; ++j) {
                    for (int i = -1; i <= 1; ++i) {
                        if (y + j >= 0 && y + j < edgeImage.rows && x + i >= 0 && x + i < edgeImage.cols) {
                            if (edgeImage.at<uchar>(y + j, x + i) >= highThreshold) {
                                result.at<uchar>(y, x) = 255; // ���� ������ ����� ���� ����
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

// �� �ߺ� ���� �Լ�
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

// ���� �� ��ȯ �Լ�
vector<Vec3f> applyHoughCircleTransform(const Mat& edgeImage, int minRadius, int maxRadius, int threshold) {
    int rows = edgeImage.rows;
    int cols = edgeImage.cols;
    vector<Vec3f> circles;

    // �� �������� ���� ���� �迭
    for (int radius = minRadius; radius <= maxRadius; radius++) {
        Mat accumulator = Mat::zeros(rows, cols, CV_32S);

        // ���� �ȼ��� ���� ���� ��ȯ ����
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (edgeImage.at<uchar>(y, x) > 0) { // ���� �ȼ� �˻�
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

        // �Ӱ谪 �̻��� �������� ���� �� ã��
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (accumulator.at<int>(y, x) > threshold) {
                    circles.push_back(Vec3f(x, y, radius));
                }
            }
        }
    }

    // ���ִ� ���� ����
    float distanceThreshold = 10.0f; // �߽��� �� �ּ� �Ÿ�
    float radiusThreshold = 5.0f; // ������ ���� �Ӱ谪
    return nonMaximumSuppressionCircles(circles, distanceThreshold, radiusThreshold);
}

int main() {
    // �̹����� �׷��̽����Ϸ� �ε�
    Mat image = imread("./canyy edge/hurfone/smarties2.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Image cannot be loaded." << endl;
        return -1;
    }
    Mat imagetext = drawPixelValues(image);
    imwrite("./canyy edge/hurfone/sobel_image_text.jpg", imagetext); // ����

    // ���� ���� �Һ� ���͸� �����Ͽ� �̹��� ���� (8��Ʈ �̹�����)
    Mat sobelHorizontalImage = applySobelHorizontalFilter8U(image);
    imwrite("./canyy edge/hurfone/sobel_horizontal.jpg", sobelHorizontalImage); // ����

    // ���� ���� �Һ� ���͸� �����Ͽ� �̹��� ���� (8��Ʈ �̹�����)
    Mat sobelVerticalImage = applySobelVerticalFilter8U(image);
    imwrite("./canyy edge/hurfone/sobel_vertical.jpg", sobelVerticalImage); // ����

    // �ؽ�Ʈȭ�� �̹��� ���� �� ����
    Mat sobelHorizontalTextImage = drawPixelValues(sobelHorizontalImage);
    Mat sobelVerticalTextImage = drawPixelValues(sobelVerticalImage);
    imwrite("./canyy edge/hurfone/sobel_horizontal_text.jpg", sobelHorizontalTextImage);
    imwrite("./canyy edge/hurfone/sobel_vertical_text.jpg", sobelVerticalTextImage);

    // ���� ���� ��� �� ����
    Mat edgeIntensityImage = calculateEdgeIntensity8U(sobelHorizontalImage, sobelVerticalImage);
    imwrite("./canyy edge/hurfone/edge_intensity.jpg", edgeIntensityImage); // ����

    // ���� ���� �ؽ�Ʈȭ�� �̹��� ���� �� ����
    Mat edgeIntensityTextImage = drawPixelValues(edgeIntensityImage);
    imwrite("./canyy edge/hurfone/edge_intensity_text.jpg", edgeIntensityTextImage);


    // �׷����Ʈ ���� ���
    Mat gradientDirection = calculateGradientDirection(sobelHorizontalImage, sobelVerticalImage);

    // ���ִ� ���� ����
    Mat nonMaxSuppressedImage = nonMaximumSuppression(edgeIntensityImage, gradientDirection);
    imwrite("./canyy edge/hurfone/non_max_suppressed.jpg", nonMaxSuppressedImage);

    // Hysteresis Thresholding ����
    uchar lowThreshold = 50;   // ���� �Ӱ谪 ����
    uchar highThreshold = 100; // ���� �Ӱ谪 ����
    Mat hysteresisImage = applyHysteresisThresholding(nonMaxSuppressedImage, lowThreshold, highThreshold);
    imwrite("./canyy edge/hurfone/hysteresis_thresholded.jpg", hysteresisImage);

    // Hough �� ��ȯ ����
    int minRadius = 10;   // �ּ� ������ ����
    int maxRadius = 100;  // �ִ� ������ ����
    int threshold = 150;   // �Ӱ谪 ����
    vector<Vec3f> circles = applyHoughCircleTransform(hysteresisImage, minRadius, maxRadius, threshold);

    // �� ǥ��
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        circle(hysteresisImage, Point(c[0], c[1]), c[2], Scalar(255, 255, 255), LINE_AA);
    }

    // ��� �̹��� ����
    imwrite("./canyy edge/hurfone/detected_circles.jpg", hysteresisImage);

    return 0;
}
