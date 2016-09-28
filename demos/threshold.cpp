#include <cmath>

#include "../src/base.hpp"
#include "../src/util.hpp"
#include "../src/threshold.hpp"

int main(int argc, char* argv[])
{
	int cameraIndex = 1;

	VideoCapture cap{cameraIndex};

	if (!cap.isOpened()) {
		std::cerr << "error: couldn't capture camera number " << cameraIndex << '\n';
		return -1;
	}

	cap.set(cv::CAP_PROP_FPS, 60);

	int mode = 7;
	bool paused = false;
	bool myThreshold = true;

	int pressedKey = 0;

	std::vector<int> colors;

	Mat frame;

	while (pressedKey != 'q') {
		if (!paused) {
			cap >> frame;
		}

		Mat grayscale;
		cv::cvtColor(frame, grayscale, CV_BGR2GRAY);

		if (myThreshold)
		{
			/*
			int minDiff = 10;
			Mat mipmap = grayscale.clone();
			cv::pyrDown(mipmap, mipmap);
			cv::blur(mipmap, mipmap, Size{3, 3});
			cv::pyrDown(mipmap, mipmap);
			cv::blur(mipmap, mipmap, Size{3, 3});
			cv::pyrDown(mipmap, mipmap);
			cv::blur(mipmap, mipmap, Size{3, 3});

			Mat foundColors{frame.rows, frame.cols, CV_8U, 127};

			int level = 3;
			int maxLevel = std::log2(std::min(frame.rows, frame.cols));
			std::cout << maxLevel;

			while (level <= maxLevel) {
				for (int r = 0; r < frame.rows; r++) {
					for (int c = 0; c < frame.cols; c++) {
						u8 &curr = foundColors.at<u8>(r, c);
						if (curr == 127) {
							u8 original = grayscale.at<u8>(r, c);
							u8 thisMip = mipmap.at<u8>(r >> level, c >> level);
							if (original >= thisMip + minDiff) {
								curr = 255;
							}
							else if (original <= thisMip - minDiff) {
								curr = 0;
							}
						}
					}
				}

				cv::pyrDown(mipmap, mipmap);
				cv::blur(mipmap, mipmap, Size{3, 3});
				level++;
			}

			cv::imshow("w", foundColors);
			*/

			Mat binary;
			leshenkoThreshold(grayscale, binary);
			cv::imshow("w", binary);
		}
		else {
			if (mode == 1) {
				cv::imshow("w", grayscale);
			}

			Mat blurred;
			cv::blur(grayscale, blurred, Size{13, 13});

			if (mode == 2) {
				cv::imshow("w", blurred);
			}

			Mat largeDiff;
			grayscale.convertTo(largeDiff, CV_16U);
			largeDiff += 127;
			Mat diff;
			cv::subtract(largeDiff, blurred, diff, cv::noArray(), CV_8U);

			if (mode == 3) {
				cv::imshow("w", diff);
			}


			if (mode == 4) {
				Mat binary;
				cv::threshold(diff, binary, 127, 255, cv::THRESH_BINARY);
				cv::imshow("w", binary);
			}

			Mat black;
			Mat white;

			int minDiff = 10;

			cv::threshold(diff, white, 127 + minDiff, 255, cv::THRESH_BINARY);
			cv::threshold(diff, black, 127 - minDiff, 255, cv::THRESH_BINARY_INV);

			Mat foundColors{frame.rows, frame.cols, CV_8U, Scalar{127}};
			cv::bitwise_or(foundColors, white, foundColors);
			cv::subtract(foundColors, black, foundColors);

			if (mode == 5) {
				cv::imshow("w", foundColors);
			}

			Mat gray{frame.rows, frame.cols, CV_8U, 255};
			gray -= black;
			gray -= white;

			if (mode == 6) {
				cv::imshow("w", gray);
			}

			Mat grayComps;
			int grayN = cv::connectedComponents(gray, grayComps, 4, CV_32S);

			colors.resize(grayN);
			std::fill(colors.begin(), colors.end(), 0);

			for (int r = 1; r < grayComps.rows; r++) {
				s32 *elem = grayComps.ptr<s32>(r);

				for (int c = 1; c < grayComps.cols; c++) {
					if (*elem != 0) {
						u8 leftColor = foundColors.at<u8>(r, c - 1);
						if (leftColor != 127) {
							colors[*elem] += leftColor ? 1 : -1;
						}

						u8 upColor = foundColors.at<u8>(r - 1, c);
						if (upColor != 127) {
							colors[*elem] += upColor ? 1 : -1;
						}
					}

					elem++;
				}
			}

			for (auto &neighbors : colors) {
				neighbors = neighbors >= 0 ? 255 : 0;
			}

			Mat binary;
			binary.create(frame.rows, frame.cols, CV_8U);

			for (int r = 0; r < binary.rows; r++) {
				u8 *p = binary.ptr<u8>(r);
				for (int c = 0; c < binary.cols; c++) {
					u8 original = foundColors.at<u8>(r, c);
					if (original != 127) {
						*p = original;
					}
					else {
						*p = colors[grayComps.at<s32>(r, c)];
					}
					p++;
				}
			}

			if (mode == 7) {
				cv::imshow("w", binary);
			}
		}

		showFrameRateInTitle("w");
		pressedKey = cv::waitKey(1);

		if (pressedKey >= '1' && pressedKey <= '9') {
			mode = pressedKey - '0';
		}
		else if (pressedKey == ' ') {
			paused = !paused;
		}
		else if (pressedKey == 'l') {
			myThreshold = !myThreshold;
		}
	}

	return 0;
}
