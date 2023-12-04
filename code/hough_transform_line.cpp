#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
// ���ִ� ������ ������ ���� ��ȯ �Լ�
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
    // �̹����� �׷��̽����Ϸ� �ҷ�����
    cv::Mat image = cv::imread("./canyy edge/hurf/hysteresis_thresholded.jpg", cv::IMREAD_GRAYSCALE); // ĳ�� ���� �� ����� �̹��� ���
    if (image.empty()) {
        std::cout << "�̹����� �ҷ����� �� �����߽��ϴ�." << std::endl;
        return -1;
    }

    int width = image.cols; // �̹��� �ʺ�
    int height = image.rows; // �̹��� ����

    // ����� ���� ���� ��ȯ ����
    std::vector<cv::Vec2f> lines;
    HoughTransform(image, lines, 1, 1, 250, 10); // rho �ػ�, theta �ػ�, �Ӱ谪 ����

    // ����� ������ �̹����� �׸���
    cv::Mat color_image;
    cv::cvtColor(image, color_image, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a * rho, y0 = b * rho;

        // ������ ���� ���� ������ ��ǥ x, y ���
        int x = static_cast<int>(x0 + 1000 * (-b));
        int y = static_cast<int>(y0 + 1000 * (a));

        // ����� �ָ� ������ �ִ� �� �� ���
        cv::Point pt1(x + 2000 * (-b), y + 2000 * (a));
        cv::Point pt2(x - 2000 * (-b), y - 2000 * (a));

        // ������ �Ǽ����� ���� �׸���
        cv::line(color_image, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // ��� �̹��� ����
    cv::imwrite("./canyy edge/hurf/hough_transform_custom_result.jpg", color_image);

    return 0;
}
