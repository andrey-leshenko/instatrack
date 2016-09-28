#include <iostream>
#include <cmath>
#include <stdint.h>

#include <opencv2/opencv.hpp>

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

using cv::Mat;
using cv::InputArray;
using cv::OutputArray;
using cv::Size;
using cv::VideoCapture;
using cv::Mat;

void showFrameRateInTitle(const char* window)
{
	static int64 freq = static_cast<int>(cv::getTickFrequency());
	static int64 captureLength = freq / 10;

	static int64 start = cv::getTickCount();
	static int frames = 0;

	frames++;

	int64 curr = cv::getTickCount();

	if ((curr - start) >= captureLength) {
		int fps = frames * (freq / (curr - start));
		cv::setWindowTitle(window, std::to_string(fps));
		start = curr;
		frames = 0;
	}
}

void leshenkoThreshold(InputArray _src, OutputArray _dst, int minDiff = 10)
{
	Mat src = _src.getMat();

	CV_Assert(src.type() == CV_8U);

	Size blurKernel{3, 3};

	int minSize = std::min(src.rows, src.cols);
	int downscalingLevels = static_cast<int>(std::log2(minSize));

	Mat foundColors = Mat::zeros(src.rows, src.cols, CV_8U);

	Mat mipmap = src.clone();
	int level = 0;

	for (int i = 0; i < 3; i++) {
		cv::pyrDown(mipmap, mipmap);
		cv::blur(mipmap, mipmap, blurKernel);
		level++;
	}

	while (level <= downscalingLevels) {
		for (int r = 0; r < src.rows; r++) {
			for (int c = 0; c < src.cols; c++) {
				u8 &curr = foundColors.at<u8>(r, c);
				if (curr == 0) {
					u8 original = src.at<u8>(r, c);
					u8 thisMip = mipmap.at<u8>(r >> level, c >> level);
					if (original >= thisMip + minDiff) {
						curr = 255;
					}
					else if (original <= thisMip - minDiff) {
						curr = 127;
					}
				}
			}
		}

		cv::pyrDown(mipmap, mipmap);
		cv::blur(mipmap, mipmap, blurKernel);
		level++;
	}

	cv::threshold(foundColors, _dst, 128, 255, cv::THRESH_BINARY);
}

void threshold2(InputArray _src, OutputArray _dst, int minDiff = 10)
{
	Mat src = _src.getMat();

	CV_Assert(src.type() == CV_8U);

	int minSize = std::min(src.rows, src.cols);
	int downscalingLevels = static_cast<int>(std::log2(minSize));

	Mat foundColors{src.rows, src.cols, CV_8U, 127};

	Mat mipmap = src.clone();
	int level = 0;

	for (int i = 0; i < 3; i++) {
		// TODO(Andrey): Gaussian blur
		cv::blur(src, mipmap, Size{1 << level, 1 << level});
		level++;
	}

	while (level <= downscalingLevels) {
		for (int r = 0; r < src.rows; r++) {
			for (int c = 0; c < src.cols; c++) {
				u8 &curr = foundColors.at<u8>(r, c);
				if (curr == 127) {
					u8 original = src.at<u8>(r, c);
					u8 thisMip = mipmap.at<u8>(r, c);
					if (original >= thisMip + minDiff) {
						curr = 255;
					}
					else if (original <= thisMip - minDiff) {
						curr = 0;
					}
				}
			}
		}

		cv::blur(src, mipmap, Size{1 << level, 1 << level});
		level++;
	}

	foundColors.copyTo(_dst);
}

int main(int argc, char* argv[])
{
	int cameraIndex = 1;
	VideoCapture cap{cameraIndex};

	if (!cap.isOpened()) {
		std::cerr << "error: couldn't capture camera number " << cameraIndex << '\n';
		return -1;
	}

	cap.set(cv::CAP_PROP_FPS, 60);

	int pressedKey = 0;

	while (pressedKey != 'q') {
		Mat frame;
		cap >> frame;

		Mat grayscale;
		cv::cvtColor(frame, grayscale, CV_BGR2GRAY);

		Mat binary;

		//leshenkoThreshold(grayscale, binary);
		threshold2(grayscale, binary);

		cv::imshow("w", binary);
		showFrameRateInTitle("w");
		pressedKey = cv::waitKey(1);
	}

	return 0;
}
