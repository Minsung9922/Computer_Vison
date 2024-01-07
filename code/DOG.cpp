#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

vector<vector<Mat>> generateMultiScaleImages(const Mat& originalImage, double initialSigma, int numOctaves, int s) {
    vector<vector<Mat>> octaves;
    Mat currentImage;
    resize(originalImage, currentImage, Size(), 2.0, 2.0, INTER_LINEAR);

    double k = pow(2.0, 1.0 / s);

    for (int octave = 0; octave < numOctaves; ++octave) {
        vector<Mat> imagesInOctave;

        if (octave == 0) {
            GaussianBlur(currentImage, currentImage, Size(), initialSigma);
        }

        imagesInOctave.push_back(currentImage.clone());

        double sigma = initialSigma;
        for (int i = 1; i < s + 3; ++i) {
            sigma *= k;
            Mat blurred;
            GaussianBlur(currentImage, blurred, Size(), sigma);
            imagesInOctave.push_back(blurred);
        }

        if (octave < numOctaves - 1) {
            resize(imagesInOctave[s], currentImage, Size(), 0.5, 0.5, INTER_LINEAR);
        }

        octaves.push_back(imagesInOctave);

        // 옥타브 내의 이미지 저장
        for (size_t i = 0; i < imagesInOctave.size(); ++i) {
            string octaveImageFilename = "Octaves/octave_" + to_string(octave) + "_image_" + to_string(i) + ".jpg";
            imwrite(octaveImageFilename, imagesInOctave[i]);
        }
    }

    return octaves;
}

vector<vector<Mat>> generateDoGImages(const vector<vector<Mat>>& octaves) {
    vector<vector<Mat>> dogImages;
    for (const auto& octave : octaves) {
        vector<Mat> dogInOctave;
        for (size_t i = 0; i < octave.size() - 1; ++i) {
            Mat dog;
            subtract(octave[i + 1], octave[i], dog, noArray(), CV_32F); // DOG 계산

            // 결과 이미지 정규화 및 값 범위 조정
            double minVal, maxVal;
            minMaxLoc(dog, &minVal, &maxVal);
            dog = (dog - minVal) / (maxVal - minVal) * 255.0;
            dog.convertTo(dog, CV_8U);

            dogInOctave.push_back(dog);

            // DoG 이미지 저장
            string dogFilename = "DoG/octave_" + to_string(dogImages.size()) + "_image_" + to_string(i) + ".jpg";
            imwrite(dogFilename, dog);
        }
        dogImages.push_back(dogInOctave);
    }
    return dogImages;
}


bool isKeyPoint(const Mat& currentImage, const Mat& prevImage, const Mat& nextImage, int x, int y) {
    // 중심 화소의 값을 가져옵니다.
    float centerPixel = currentImage.at<uchar>(y, x); // 타입을 uchar로 수정

    // 주변 8개 화소와 이웃한 영상의 9개 화소를 비교합니다.
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            // 중심 화소와의 비교는 제외합니다.
            if (dx == 0 && dy == 0) {
                continue;
            }

            // 주변 화소의 값을 가져옵니다.
            float neighborPixel = currentImage.at<uchar>(y + dy, x + dx); // 타입을 uchar로 수정
            float neighborPrevPixel = prevImage.at<uchar>(y + dy, x + dx); // 타입을 uchar로 수정
            float neighborNextPixel = nextImage.at<uchar>(y + dy, x + dx); // 타입을 uchar로 수정

            // 중심 화소가 주변 화소보다 작으면 극점이 아닙니다.
            if (centerPixel <= neighborPixel ||
                centerPixel <= neighborPrevPixel ||
                centerPixel <= neighborNextPixel) {
                return false;
            }
        }
    }

    // 모든 비교에서 중심 화소가 주변 화소보다 크면 극점으로 판단합니다.
    return true;
}

vector<vector<vector<KeyPoint>>> findKeyPoints(const vector<vector<Mat>>& dogImages) {
    vector<vector<vector<KeyPoint>>> allKeypoints;
    for (size_t octave = 0; octave < dogImages.size(); ++octave) {
        vector<vector<KeyPoint>> octaveKeypoints;
        for (size_t i = 1; i < dogImages[octave].size() - 1; ++i) { // 2번째부터 마지막 - 1번째까지의 DoG 영상 사용
            const Mat& prevImage = dogImages[octave][i - 1];
            const Mat& currentImage = dogImages[octave][i];
            const Mat& nextImage = dogImages[octave][i + 1];

            vector<KeyPoint> keypoints;
            for (int y = 1; y < currentImage.rows - 1; ++y) {
                for (int x = 1; x < currentImage.cols - 1; ++x) {
                    if (isKeyPoint(currentImage, prevImage, nextImage, x, y)) {
                        keypoints.push_back(KeyPoint(x, y, octave));
                    }
                }
            }
            octaveKeypoints.push_back(keypoints);
        }
        allKeypoints.push_back(octaveKeypoints);
    }
    return allKeypoints;
}


int main() {
    Mat originalImage = imread("SIFT/apple_gray.jpg", IMREAD_GRAYSCALE); // 이미지를 그레이스케일로 읽음
    vector<vector<Mat>> octaves = generateMultiScaleImages(originalImage, 1.6, 4, 3);

    vector<vector<Mat>> dogImages = generateDoGImages(octaves);

    vector<vector<vector<KeyPoint>>> allKeypoints = findKeyPoints(dogImages);

    // 옥타브 내의 이미지 저장
    for (size_t octave = 0; octave < octaves.size(); ++octave) {
        for (size_t i = 0; i < octaves[octave].size(); ++i) {
            string octaveImageFilename = "SIFT/octave_" + to_string(octave) + "_image_" + to_string(i) + ".jpg";
            imwrite(octaveImageFilename, octaves[octave][i]);
        }
    }

    // 특징점 이미지 저장
    for (size_t octave = 0; octave < allKeypoints.size(); ++octave) {
        for (size_t i = 0; i < allKeypoints[octave].size(); ++i) {
            // 현재 옥타브의 이미지 해상도에 맞게 이미지 크기를 조정
            Mat imageWithKeypoints;
            resize(originalImage, imageWithKeypoints, octaves[octave][i + 1].size());

            // 배경을 검정색으로 설정
            imageWithKeypoints.setTo(Scalar(0));

            for (size_t j = 0; j < allKeypoints[octave][i].size(); ++j) {
                int x = static_cast<int>(allKeypoints[octave][i][j].pt.x);
                int y = static_cast<int>(allKeypoints[octave][i][j].pt.y);
                circle(imageWithKeypoints, Point(x, y), 2, Scalar(255), -1); // 특징점을 하얀색으로 표시
            }

            string keypointsFilename = "SIFT/keypoints_octave_" + to_string(octave) + "_image_" + to_string(i) + ".jpg";
            imwrite(keypointsFilename, imageWithKeypoints);
        }
    }

    return 0;
}
