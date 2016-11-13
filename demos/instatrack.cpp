#include "../src/instatrack.hpp"

int main(int argc, char *argv[])
{
	int cameraIndex = 0;
	cv::Size chessSize{5, 4};

	if (argc - 1 > 0) {
		cameraIndex = atoi(argv[1]);
	}

	return instatrackDemo(cameraIndex, chessSize);
}
