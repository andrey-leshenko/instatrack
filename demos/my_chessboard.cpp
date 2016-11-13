#include "base.hpp"
#include "util.hpp"
#include "../src/instatrack.hpp"


int main(int argc, char *argv[])
{
	int cameraIndex = 0;
	Size chessboardSize{5, 4};

	if (argc - 1 > 0) {
		cameraIndex = atoi(argv[1]);
	}

	VideoCapture cam{cameraIndex};

	if (!cam.isOpened()) {
		std::cerr << "Error: Couldn't capture camera." << std::endl;
		return -1;
	}

	cam.set(cv::CAP_PROP_FPS, 60);

	Mat frame;
	vector<Point2f> chessPoints;

	int pressedKey = 0;
	bool searching = true;
	bool adaptiveThreshold = false;

	cam.grab();

	while (pressedKey != 'q') {
		cam.retrieve(frame);
		cam.grab();

		if (searching) {
			int flags = 0;
			if (adaptiveThreshold) {
				flags |= CV_CALIB_CB_ADAPTIVE_THRESH;
			}
			bool found = instatrackFindChessboard(
					frame,
					chessboardSize,
					chessPoints,
					flags);

			drawChessboardCorners(frame, chessboardSize, chessPoints, found);
		}

		imshow("w", frame);
		showFrameRateInTitle("w");
		pressedKey = cv::waitKey(1);

		if (pressedKey == ' ') {
			searching = !searching;
		}
		else if (pressedKey == 'a') {
			adaptiveThreshold = !adaptiveThreshold;
		}
	}

	return 0;
}
