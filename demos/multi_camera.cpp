#include "base.hpp"
#include "util.hpp"

#include <vector>

using std::vector;

int main(int argc, char *argv[])
{
	vector<VideoCapture> cams {1, 2};
	CV_Assert(cams.size() == 2);
	vector<Mat> frame(cams.size());

	for (auto c : cams) {
		if (!c.isOpened()) {
			std::cerr << "Error: Couldn't capture camera." << std::endl;
			return -1;
		}
	}

	Mat display;

	int targetFPS = 30;
	int maxFPS = 150;

	cv::namedWindow("w", CV_WINDOW_AUTOSIZE);
	cv::createTrackbar("Target FPS: ", "w", &targetFPS, maxFPS);

	int pressedKey = 0;

	while (pressedKey != 'q') {
		printTimeSinceLastCall("start\n");
		for (int i = 0; i < cams.size(); i++) {
			cams[i] >> frame[i];
			printTimeSinceLastCall("cap");
		}

		cv::hconcat(frame[0], frame[1], display);

		cv::imshow("w", display);
		showFrameRateInTitle("w");

		if (pressedKey == ' ') {
			for (auto c : cams) {
				c.set(cv::CAP_PROP_FPS, targetFPS);
			}
		}

		printTimeSinceLastCall("finish");

		pressedKey = cv::waitKey(1);
	}

	return 0;
}
