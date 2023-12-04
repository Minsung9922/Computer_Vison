#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
// 비최대 억제를 적용한 허프 변환 함수
void HoughTransform(const cv::Mat& edges, std::vector<cv::Vec2f>& lines, int rho_resolution, double theta_resolution, int threshold, int nms_distance) {
    int width = edges.cols;
    int height = edges.rows;
    double max_dist = std::sqrt(width * width + height * height);

    int num_thetas = 180 / theta_resolution;
    int num_rhos = static_cast<int>(2 * max_dist / rho_resolution);

    cv::Mat accum = cv::Mat::zeros(num_rhos, num_thetas, CV_32S);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                for (int theta_idx = 0; theta_idx < num_thetas; theta_idx++) {
                    double theta = theta_idx * theta_resolution - 90.0;
                    double rho = x * std::cos(theta * CV_PI / 180.0) + y * std::sin(theta * CV_PI / 180.0);
                    int rho_idx = static_cast<int>((rho + max_dist) / rho_resolution);
                    accum.at<int>(rho_idx, theta_idx)++;
                }
            }
        }
    }

    for (int rho_idx = 0; rho_idx < num_rhos; rho_idx++) {
        for (int theta_idx = 0; theta_idx < num_thetas; theta_idx++) {
            int current_accum = accum.at<int>(rho_idx, theta_idx);
            if (current_accum >= threshold) {
                double rho = rho_idx * rho_resolution - max_dist;
                double theta = theta_idx * theta_resolution - 90.0;

                bool is_maximum = true;
                for (int y = -nms_distance; y <= nms_distance; y++) {
                    for (int x = -nms_distance; x <= nms_distance; x++) {
                        if (y + rho_idx >= 0 && y + rho_idx < num_rhos && x + theta_idx >= 0 && x + theta_idx < num_thetas) {
                            int neighbor_accum = accum.at<int>(y + rho_idx, x + theta_idx);
                            if (neighbor_accum > current_accum) {
                                is_maximum = false;
                                break;
                            }
                        }
                    }
                    if (!is_maximum) {
                        break;
                    }
                }

                if (is_maximum) {
                    lines.push_back(cv::Vec2f(static_cast<float>(rho), static_cast<float>(theta * CV_PI / 180.0)));
                }
            }
        }
    }
}
int main() {
    // 이미지를 그레이스케일로 불러오기
    cv::Mat image = cv::imread("./canyy edge/hurf/hysteresis_thresholded.jpg", cv::IMREAD_GRAYSCALE); // 캐니 에지 후 검출된 이미지 사용
    if (image.empty()) {
        std::cout << "이미지를 불러오는 데 실패했습니다." << std::endl;
        return -1;
    }

    int width = image.cols; // 이미지 너비
    int height = image.rows; // 이미지 높이

    // 사용자 정의 허프 변환 적용
    std::vector<cv::Vec2f> lines;
    HoughTransform(image, lines, 1, 1, 250, 10); // rho 해상도, theta 해상도, 임계값 조정

    // 검출된 직선을 이미지에 그리기
    cv::Mat color_image;
    cv::cvtColor(image, color_image, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a * rho, y0 = b * rho;

        // 수선의 발을 내려 만나는 좌표 x, y 계산
        int x = static_cast<int>(x0 + 1000 * (-b));
        int y = static_cast<int>(y0 + 1000 * (a));

        // 충분히 멀리 떨어져 있는 두 점 계산
        cv::Point pt1(x + 2000 * (-b), y + 2000 * (a));
        cv::Point pt2(x - 2000 * (-b), y - 2000 * (a));

        // 빨간색 실선으로 선분 그리기
        cv::line(color_image, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // 결과 이미지 저장
    cv::imwrite("./canyy edge/hurf/hough_transform_custom_result.jpg", color_image);

    return 0;
}
