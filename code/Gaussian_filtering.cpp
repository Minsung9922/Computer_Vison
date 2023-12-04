#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace cv;
using namespace std;

// ����þ� Ŀ���� ����ϴ� �Լ�
void calculateGaussianKernel(vector<vector<double>>& kernel, int kernelSize, double sigma) {
    const int halfSize = kernelSize / 2;
    double sum = 0.0; // Ŀ���� ���� ����ϱ� ���� ����
    kernel.resize(kernelSize, vector<double>(kernelSize, 0));

    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            kernel[i + halfSize][j + halfSize] = exp(-(i * i + j * j) / (2 * sigma * sigma));
            sum += kernel[i + halfSize][j + halfSize];
        }
    }

    // Ŀ�� ����ȭ
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel[i][j] /= sum;
        }
    }
}

// �̹����� �е��� �߰��ϴ� �Լ�
Mat addPadding(const Mat& inputImage, int padSize) {
    Mat paddedImage(inputImage.rows + 2 * padSize, inputImage.cols + 2 * padSize, inputImage.type(), Scalar::all(0));

    // �Է� �̹����� ���ο� �е� �̹����� �߾ӿ� ����
    inputImage.copyTo(paddedImage(Rect(padSize, padSize, inputImage.cols, inputImage.rows)));

    // �׵θ� ���縦 ���� �е� �߰�
    for (int i = 0; i < padSize; ++i) {
        // ��� �е�
        paddedImage.row(padSize).copyTo(paddedImage.row(i));
        // �ϴ� �е�
        paddedImage.row(paddedImage.rows - padSize - 1).copyTo(paddedImage.row(paddedImage.rows - i - 1));
        // ���� �е�
        paddedImage.col(padSize).copyTo(paddedImage.col(i));
        // ���� �е�
        paddedImage.col(paddedImage.cols - padSize - 1).copyTo(paddedImage.col(paddedImage.cols - i - 1));
    }

    return paddedImage;
}

// ����þ� ���͸� �̹����� �����ϴ� �Լ�
void applyGaussianFilter(const Mat& inputImage, Mat& outputImage, int kernelSize, double sigma) {
    const int halfSize = kernelSize / 2;
    vector<vector<double>> kernel;

    // ����þ� Ŀ�� ���
    calculateGaussianKernel(kernel, kernelSize, sigma);

    // �Է� �̹����� �е� �߰�
    Mat paddedImage = addPadding(inputImage, halfSize);

    // ��� �̹��� �ʱ�ȭ
    outputImage = Mat::zeros(inputImage.size(), inputImage.type());

    // �е��� �̹����� ����þ� ���� ����
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            double filteredPixel = 0.0;

            for (int i = -halfSize; i <= halfSize; ++i) {
                for (int j = -halfSize; j <= halfSize; ++j) {
                    // Ŀ�ΰ� �̹����� �����ϴ� �ȼ����� ���Ͽ� ������
                    filteredPixel += kernel[i + halfSize][j + halfSize] * paddedImage.at<uchar>(y + i + halfSize, x + j + halfSize);
                }
            }

            outputImage.at<uchar>(y, x) = saturate_cast<uchar>(filteredPixel);
        }
    }
}

int main() {
    // �̹����� �׷��̽����Ϸ� �ҷ�����
    Mat image = imread("apple.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "�̹����� �ҷ��� �� �����ϴ�." << endl;
        return -1;
    }

    // ���͸��� �̹����� ������ ��ü
    Mat filteredImage;

    // ����� Ŀ���� ũ��� �ñ׸� ���� ����
    const int kernelSize = 5;
    const double sigma = 1.0;

    // ����þ� ���͸� ���� �����ϴ� �Լ� ȣ��
    applyGaussianFilter(image, filteredImage, kernelSize, sigma);

    // ���͸��� �̹����� ����
    imwrite("apple_filtered_sigma_2.jpg", filteredImage);

    cout << "����þ� ���Ͱ� ����� �̹����� �����߽��ϴ�." << endl;

    return 0;
}
