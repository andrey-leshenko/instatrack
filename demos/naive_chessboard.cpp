#include "base.hpp"
#include "util.hpp"


int main(int argc, char *argv[])
{
	int cameraIndex = 0;
	Size chessboardSize{8, 5};

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
			bool found = cv::findChessboardCorners(
					frame,
					chessboardSize,
					chessPoints,
					CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FAST_CHECK);

			drawChessboardCorners(frame, chessboardSize, chessPoints, found);
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
