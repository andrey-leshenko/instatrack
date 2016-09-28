#include "../src/base.hpp"
#include "../src/util.hpp"
#include "../src/chessboard.hpp"

const Size chessboardSize{7, 4};

int main()
{
	VideoCapture cam{1};

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
			bool found = findChessboardSquares(
					frame,
					chessboardSize,
					chessPoints);

			std::cout << found << std::endl;
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
