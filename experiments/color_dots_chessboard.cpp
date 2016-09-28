#include <iostream>
#include <stdint.h>
#include <vector>

#include <opencv2/opencv.hpp>

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

using std::vector;

using cv::VideoCapture;
using cv::Mat;
using cv::Scalar;
using cv::InputArray;
using cv::OutputArray;

#define QQQ do {std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;} while(0)

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

struct Blob
{
	int greenNeighbors[4];
	int whiteNeighbors[4];
	int blackNeighbors[4];
	bool invalid;
};

void MarkAsNeighbor(Blob &a, u8 bColor, int bIndex)
{
	int *neighborsArray = NULL;

	switch (bColor) {
		case 0:
			neighborsArray = a.blackNeighbors;
			break;
		case 127:
			neighborsArray = a.greenNeighbors;
			break;
		case 255:
			neighborsArray = a.whiteNeighbors;
			break;
	}

	for (int i = 0; i < 4; i++) {
		if (neighborsArray[i] == bIndex) {
			return;
		}
		if (neighborsArray[i] == 0) {
			neighborsArray[i] = bIndex;
			return;
		}
	}

	a.invalid = true;
}

struct ChessComps
{
	Mat img;

	Mat allComps;

	Mat greenComps;
	Mat whiteComps;
	Mat blackComps;

	Mat greenStats;
	Mat whiteStats;
	Mat blackStats;

	Mat greenCenters;
	Mat whiteCenters;
	Mat blackCenters;

	int totalCompCount;

	vector<Blob> blobs;
};

void ChessCompsBuild(ChessComps &comps, Mat &img, InputArray greenMask, InputArray whiteMask, InputArray blackMask)
{
	comps.img = img;

	int greenN = cv::connectedComponentsWithStats(greenMask, comps.greenComps, comps.greenStats, comps.greenCenters, 4, CV_32S);
	int whiteN = cv::connectedComponentsWithStats(whiteMask, comps.whiteComps, comps.whiteStats, comps.whiteCenters, 4, CV_32S);
	int blackN = cv::connectedComponentsWithStats(blackMask, comps.blackComps, comps.blackStats, comps.whiteCenters, 4, CV_32S);

	comps.allComps = comps.blackComps;
	comps.allComps += comps.whiteComps;
	comps.allComps += comps.greenComps;

	// TODO(Andrey): whiteN - 1?
	cv::add(comps.allComps, whiteN + greenN, comps.allComps, blackMask);
	cv::add(comps.allComps, greenN, comps.allComps, whiteMask);

	Mat uc;
	comps.allComps.convertTo(uc, CV_8U);


	comps.totalCompCount = greenN + whiteN + blackN;
	if (comps.totalCompCount > comps.blobs.size()) {
		comps.blobs.resize(comps.totalCompCount);
	}

	Blob empty = {};
	comps.blobs.assign(comps.blobs.size(), empty);


	int imRows = img.rows;
	int imCols = img.cols;

	for (int r = 0; r < imRows - 1; r++) {
		for (int c = 0; c < imCols - 1; c++) {
			u8 elem = img.at<u8>(r, c);
			u8 right = img.at<u8>(r, c + 1);
			u8 down = img.at<u8>(r + 1, c);

			if (elem != right) {
				int compA = comps.allComps.at<s32>(r, c);
				int compB = comps.allComps.at<s32>(r, c + 1);
				MarkAsNeighbor(comps.blobs[compA], right, compB);
				MarkAsNeighbor(comps.blobs[compB], elem, compA);
			}
			if (elem != down) {
				int compA = comps.allComps.at<s32>(r, c);
				int compB = comps.allComps.at<s32>(r + 1, c);
				MarkAsNeighbor(comps.blobs[compA], down, compB);
				MarkAsNeighbor(comps.blobs[compB], elem, compA);
			}
		}
	}

	/*
	{
		Blob &b = comps.blobs[22];
		
		std::cout << "green: ";
		for (int i = 0; i < 4; i++) {
			std::cout << b.greenNeighbors[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "white: ";
		for (int i = 0; i < 4; i++) {
			std::cout << b.whiteNeighbors[i] << " ";
		}
		std::cout << std::endl;
		std::cout << "black: ";
		for (int i = 0; i < 4; i++) {
			std::cout << b.blackNeighbors[i] << " ";
		}
		std::cout << std::endl;
		std::cout << b.invalid << std::endl;
	}

	cv::imshow("w", uc);
	cv::waitKey(0);
	*/
}

int main(int argc, char* argv[])
{
	int cameraIndex = 1;

	VideoCapture cap;
	Mat loadedImage;

	if (cameraIndex >= 0) {
		cap = VideoCapture{cameraIndex};
		cap.set(cv::CAP_PROP_FPS, 60);

		if (!cap.isOpened()) {
			std::cerr << "error: couldn't capture camera number " << cameraIndex << '\n';
			return -1;
		}
	}
	else {
		auto fileName = "chess_color.png";
		loadedImage = cv::imread(fileName);

		if (!loadedImage.data) {
			std::cerr << "error: couldn't read file " << fileName << '\n';
			return -1;
		}
	}

	Mat frame;
	Mat green;
	Mat grayscale;
	Mat hsv;
	Mat binary;

	Mat greenMask;
	Mat blackMask;
	Mat whiteMask;

	ChessComps comps;

	int sensitivity = 20;
	int greenHue = 75;

	int pressedKey = 0;
	if (cameraIndex >= 0) {
		cap.grab();
	}

	while (pressedKey != 'q') {
		if (cameraIndex >= 0) {
			cap.retrieve(frame);
			cap.grab();
		}
		else {
			frame = loadedImage.clone();
		}

		cv::cvtColor(frame, grayscale, CV_BGR2GRAY);
		cv::cvtColor(frame, hsv, CV_BGR2HSV);

		// TODO(Andrey): Blur image before searching for colors

		cv::inRange(hsv, Scalar(greenHue - sensitivity, 100, 100), Scalar(greenHue + sensitivity, 255, 255), greenMask);

		cv::threshold(grayscale, binary, 80, 255, CV_THRESH_BINARY);
		cv::bitwise_and(binary, 0, binary, greenMask);
		binary.copyTo(whiteMask);

		cv::bitwise_xor(binary, 0xff, binary);
		cv::bitwise_and(binary, 0, binary, greenMask);

		binary.copyTo(blackMask);

		cv::bitwise_xor(binary, 0xff, binary);
		cv::bitwise_and(binary, 0, binary, greenMask);

		cv::bitwise_and(greenMask, 0x7f, green);
		binary += green;

		ChessCompsBuild(comps, binary, greenMask, whiteMask, blackMask);

		cv::imshow("Video feed", binary);
		showFrameRateInTitle("Video feed");
		pressedKey = cv::waitKey(1);
	}

	return 0;
}
