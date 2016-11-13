#include "base.hpp"
#include "util.hpp"


int main(int argc, char *argv[])
{
	int cameraIndex = 0;
	Size patternSize{6, 7};

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

	cam.grab();

	while (pressedKey != 'q') {
		cam.retrieve(frame);
		cam.grab();

		if (searching) {
			bool found = cv::findCirclesGrid(
					frame,
					patternSize,
					chessPoints,
					cv::CALIB_CB_SYMMETRIC_GRID);

			drawChessboardCorners(frame, patternSize, chessPoints, found);
		}

		imshow("w", frame);
		showFrameRateInTitle("w");
		pressedKey = cv::waitKey(1);

		if (pressedKey == ' ') {
			searching = !searching;
		}
	}

	return 0;
}
