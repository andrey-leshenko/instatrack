#include "base.hpp"
#include "util.hpp"

int main(int argc, char *argv[])
{
	int cameraIndex = 0;

	if (argc - 1 > 0) {
		cameraIndex = atoi(argv[1]);
	}

	VideoCapture cam{cameraIndex};

	if (!cam.isOpened()) {
		std::cerr << "Error: Couldn't capture camera." << std::endl;
		return -1;
	}

	cam.set(cv::CAP_PROP_FPS, 60);

	int thresh = 200;
	int max_thresh = 255;

	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	Mat frame;
	Mat harris;
	Mat harrisNorm;
	Mat harrisNormScaled;

	cv::namedWindow("w", CV_WINDOW_AUTOSIZE);
	cv::createTrackbar("Threshold: ", "w", &thresh, max_thresh);

	int pressedKey = 0;

	while (pressedKey != 'q') {
		cam >> frame;

		cv::cvtColor(frame, frame, CV_BGR2GRAY);
		harris = Mat::zeros(frame.size(), CV_32F);

		cv::cornerHarris(frame, harris, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
		harrisNorm *= 0.1;
		//harrisNorm.convertTo(harrisNormScaled, CV_8U);

		cv::normalize(harris, harrisNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, Mat());
		cv::convertScaleAbs(harrisNorm, harrisNormScaled);

		int meanIntensity = cv::mean(harrisNormScaled)[0];
		cv::threshold(harrisNormScaled, harrisNormScaled, thresh * meanIntensity / 127, 255, cv::THRESH_BINARY);
		//cv::erode(harrisNormScaled, harrisNormScaled, Mat());
		cv::dilate(harrisNormScaled, harrisNormScaled, Mat());

		for(int j = 0; false && j < harrisNormScaled.rows ; j++)
		{
			for(int i = 0; i < harrisNormScaled.cols; i++)
			{
				if((int) harrisNormScaled.at<float>(j,i) > thresh)
				{
					cv::circle(harrisNormScaled, Point(i, j), 5,  Scalar(0), 2, 8, 0);
				}
			}
		}

		cv::imshow("w", harrisNormScaled);
		showFrameRateInTitle("w");
		pressedKey = cv::waitKey(1);
	}

	return 0;
}
