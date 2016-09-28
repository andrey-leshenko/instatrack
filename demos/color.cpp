#include "../src/base.hpp"
#include "../src/util.hpp"

int main(int argc, char* argv[])
{
	int cameraIndex = 1;

	VideoCapture cap{cameraIndex};

	if (!cap.isOpened()) {
		std::cerr << "Error: Couldn't capture camera number " << cameraIndex << '\n';
		return -1;
	}

	cap.set(cv::CAP_PROP_FPS, 60);

	Mat frame;
	Mat hsv;
	Mat binary;
	Mat display;

	Mat output;

	enum class Mode {
		color,
		hue,
		saturation,
		value
	};

	Mode m = Mode::hue;

	int pressedKey = 0;

	while (pressedKey != 'q') {
		cap >> frame;

		cv::cvtColor(frame, hsv, CV_BGR2HSV);

		Mat channelsHSV[3];
		cv::split(hsv, channelsHSV);

		if (m == Mode::color) {
			cv::imshow("w", frame);
		}
		else if (m == Mode::hue) {
			channelsHSV[1] = Mat{frame.rows, frame.cols, CV_8U, Scalar{255}};
			channelsHSV[2] = Mat{frame.rows, frame.cols, CV_8U, Scalar{255}};

			cv::merge(channelsHSV, 3, output);
			cv::cvtColor(output, display, CV_HSV2BGR);

			cv::imshow("w", display);
		}
		else if (m == Mode::saturation) {
			cv::imshow("w", channelsHSV[1]);
		}
		else if (m == Mode::value) {
			cv::imshow("w", channelsHSV[2]);
		}

		showFrameRateInTitle("w");
		pressedKey = cv::waitKey(1);

		switch (pressedKey) {
			case 'c':
				m = Mode::color;
				break;
			case 'h':
				m = Mode::hue;
				break;
			case 's':
				m = Mode::saturation;
				break;
			case 'v':
				m = Mode::value;
				break;
		}
	}

	return 0;
}
