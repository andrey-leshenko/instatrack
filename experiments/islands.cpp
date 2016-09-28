#include <iostream>
#include <vector>
#include <algorithm>

#include <stdio.h>
#include <stdint.h>

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
using cv::Point2d;
using cv::Point2f;
using cv::Vec2d;
using cv::Size;
using cv::Range;

using cv::Point;

#define QQQ do {std::cerr << "QQQ " << __FUNCTION__ << " " << __LINE__ << std::endl;} while(0)

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

void printTime(const char* message)
{
	static s64 freq = static_cast<int>(cv::getTickFrequency());
	static s64 last = cv::getTickCount();

	s64 curr = cv::getTickCount();
	s64 delta = curr - last;
	double deltaMs = (double)delta / freq * 1000;
	printf("%s: %.2f\n", message, deltaMs);

	last = curr;
}

struct Blob
{
	int neighbors[4];
	bool invalid;
};

void markAsNeighbor(Blob &a, int bIndex)
{
	for (int i = 0; i < 4; i++) {
		if (a.neighbors[i] == bIndex) {
			return;
		}
		if (a.neighbors[i] == 0) {
			a.neighbors[i] = bIndex;
			return;
		}
	}

	a.invalid = true;
}

bool closeCircle(const vector<Blob> &blobs, int a, int b, int &ax, int &bx, int notAx = -1)
{
	const Blob &blobA = blobs[a];
	const Blob &blobB = blobs[b];

	for (int i = 0; i < 4; i++) {
		int maybeAx = blobA.neighbors[i];

		if (maybeAx == 0) {
			break;
		}
		if (maybeAx == b || maybeAx == notAx) {
			continue;
		}

		const Blob &blobMaybeAx = blobs[maybeAx];

		if (blobMaybeAx.invalid) {
			continue;
		}

		for (int j = 0; j < 4; j++) {
			int maybeBx = blobMaybeAx.neighbors[j];

			if (maybeBx == 0) {
				break;
			}
			if (blobs[maybeBx].invalid) {
				continue;
			}

			for (int k = 0; k < 4; k++) {
				if (blobB.neighbors[k] == 0) {
					break;
				}
				if (maybeBx == blobB.neighbors[k] &&
					maybeBx != a) {
					ax = maybeAx;
					bx = maybeBx;
					return true;
				}
			}
		}
	}

	return false;
}

bool areClockwiseOnImage(Vec2d a, Vec2d b, Vec2d c, Vec2d d)
{
	Vec2d ac = c - a;
	Vec2d bd = d - b;

	Vec2d acNormal{ac(1), -ac(0)};
	double dot = acNormal.dot(bd);

	bool clockwise = dot >= 0;
	bool clockwiseWhenYIsDown = !clockwise;

	return clockwiseWhenYIsDown;
}

// XXX
void show(InputArray _m)
{
	static Mat uc;

	Mat m = _m.getMat();
	m.convertTo(uc, CV_8U);
	cv::imshow("Video feed", uc);
	cv::waitKey(0);
}

int main(int argc, char* argv[])
{
	int cameraIndex = 1;

	//Size gridSize{3, 4};
	Size gridSize{7, 4};
	int gridSizeMin = std::min(gridSize.width, gridSize.height);
	int gridSizeMax = std::max(gridSize.width, gridSize.height);

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
		auto fileName = "../chess_01.png";
		loadedImage = cv::imread(fileName);

		if (!loadedImage.data) {
			std::cerr << "error: couldn't read file " << fileName << '\n';
			return -1;
		}
	}

	Mat frame;
	Mat grayscale;
	Mat ternary;

	Mat whiteMask;
	Mat blackMask;
	Mat whiteMaskInv;

	Mat whiteComps;
	Mat blackComps;
	Mat allComps;

	Mat whiteStats;
	Mat blackStats;

	Mat whiteCenters{100, 2, CV_64F};
	Mat blackCenters{100, 2, CV_64F};

	int totalCompCount;

	Mat erosionKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{11, 11});
	Mat dilationKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{3, 3});

	vector<Blob> blobs;

	Mat chess{gridSizeMax, gridSizeMax, CV_32S};

	int pressedKey = 0;

	if (cameraIndex >= 0) {
		cap.grab();
	}

	while (pressedKey != 'q') {
		printTime("Start");

		//
		// Capture a frame
		//

		if (cameraIndex >= 0) {
			cap.retrieve(frame);
			cap.grab();
		}
		else {
			frame = loadedImage.clone();
		}

		printTime("Capture image");

		//
		// Find ternary version
		//

		cv::cvtColor(frame, grayscale, CV_BGR2GRAY);
		//cv::GaussianBlur(grayscale, grayscale, Size{31, 31}, 0);

		if (true) {
			cv::threshold(grayscale, whiteMask, 80, 255, CV_THRESH_BINARY);
		}
		else {
			cv::adaptiveThreshold(grayscale, whiteMask, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 0);
		}
		Mat binary = whiteMask.clone();
		cv::bitwise_not(whiteMask, blackMask);

		cv::dilate(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 2);
		cv::erode(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 6);
		cv::dilate(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 1);

		cv::dilate(blackMask, blackMask, Mat{}, Point{-1, -1}, 2);
		cv::erode(blackMask, blackMask, Mat{}, Point{-1, -1}, 6);
		cv::dilate(blackMask, blackMask, Mat{}, Point{-1, -1}, 1);

		/*
		cv::dilate(whiteMask, whiteMask, dilationKernel);
		cv::erode(whiteMask, whiteMask, erosionKernel);
		cv::dilate(whiteMask, whiteMask, dilationKernel);

		cv::dilate(blackMask, blackMask, dilationKernel);
		cv::erode(blackMask, blackMask, erosionKernel);
		cv::dilate(blackMask, blackMask, dilationKernel);
		*/

		cv::bitwise_not(whiteMask, whiteMaskInv);
		cv::bitwise_and(blackMask, whiteMaskInv, blackMask);

		ternary.create(frame.rows, frame.cols, CV_8U);
		ternary.setTo(Scalar{127});
		cv::subtract(ternary, blackMask, ternary);
		cv::bitwise_or(ternary, whiteMask, ternary);

		printTime("Find ternary version");

		//
		// Find connected components
		//

		int whiteN = cv::connectedComponentsWithStats(whiteMask, whiteComps, whiteStats, whiteCenters, 4, CV_32S);
		int blackN = cv::connectedComponentsWithStats(blackMask, blackComps, blackStats, blackCenters, 4, CV_32S);
		/*
		int whiteN = cv::connectedComponents(whiteMask, whiteComps, 4, CV_32S);
		int blackN = cv::connectedComponents(blackMask, blackComps, 4, CV_32S);
		*/

		allComps = blackComps.clone();
		int foregroundBlackN = blackN - 1;
		cv::add(allComps, foregroundBlackN, allComps, whiteMask);
		allComps += whiteComps;

		int totalCompCount = whiteN + foregroundBlackN;

		printTime("Find connected components");

		//
		// Find neighbors
		//

		if (totalCompCount > blobs.size()) {
			blobs.resize(totalCompCount);
		}

		{
			Blob empty = {0};
			std::fill(blobs.begin(), blobs.end(), empty);
		}


		auto findNeighborsHorizontal = [](vector<Blob> &blobs, Mat ternary, Mat allComps)
		{
			const int maxDistance = 14;

			int imRows = ternary.rows;
			int imCols = ternary.cols;

			for (int r = 0; r < imRows; r++) {
				u8 lastColor = 127;
				s32 lastComp = 0;
				int distance = 0;

				for (int c = 0; c < imCols; c++) {
					u8 currColor = ternary.at<u8>(r, c);
					s32 currComp = allComps.at<s32>(r, c);

					if (currComp == 0) {
						distance++;
					}
					else if (currComp == lastComp) {
						distance = 1;
					}
					else {
						if (lastComp != 0 && lastColor != currColor && distance <= maxDistance) {
							markAsNeighbor(blobs[lastComp], currComp);
							markAsNeighbor(blobs[currComp], lastComp);
						}

						lastColor = currColor;
						lastComp = currComp;
						distance = 1;
					}
				}
			}
		};

		{
			findNeighborsHorizontal(blobs, ternary, allComps);
			Mat ternaryT = ternary.t();
			Mat allCompsT = allComps.t();
			findNeighborsHorizontal(blobs, ternaryT, allCompsT);
		}

		printTime("Find neighbors");

		//
		// Visualize neighbors graph
		//

		for (int i = 1; i < whiteN; i++) {
			Point2d center{whiteCenters.at<double>(i, 0), whiteCenters.at<double>(i, 1)};
			cv::circle(ternary, center, 5, Scalar{180, 10, 10}, -1);
		}

		for (int i = 1; i < blackN; i++) {
			Point2d center{blackCenters.at<double>(i, 0), blackCenters.at<double>(i, 1)};
			cv::circle(ternary, center, 5, Scalar{180, 10, 10}, -1);
		}

		for (int i = 1; i < whiteN; i++) {
			Blob b = blobs[i + foregroundBlackN];
			Point2d p1{whiteCenters.at<double>(i, 0), whiteCenters.at<double>(i, 1)};

			for (int k = 0; k < 4; k++) {
				int n = b.neighbors[k];

				if (n == 0) {
					break;
				}

				Point2d p2{blackCenters.at<double>(n, 0), blackCenters.at<double>(n, 1)};

				cv::line(ternary, p1, p2, Scalar{180, 10, 10});
			}
		}

		for (int i = 1; i < blackN; i++) {
			Blob b = blobs[i];
			Point2d p1{blackCenters.at<double>(i, 0), blackCenters.at<double>(i, 1)};

			for (int k = 0; k < 4; k++) {
				int n = b.neighbors[k];

				if (n == 0) {
					break;
				}

				n -= foregroundBlackN;

				Point2d p2{whiteCenters.at<double>(n, 0), whiteCenters.at<double>(n, 1)};

				cv::line(ternary, p1, p2, Scalar{180, 10, 10});
			}
		}

		//
		// Find the grid neighborhood
		//

		CV_Assert(gridSize.height > 1 && gridSize.width > 1);

		for (int i = 1; i < blackN; i++) {
			Blob &b = blobs[i];
			if (b.neighbors[3] || !b.neighbors[2]) {
				continue;
			}

			chess.setTo(0);
			chess.at<s32>(0, 0) = i;
			chess.at<s32>(0, 1) = blobs[b.neighbors[0]].invalid ?
				b.neighbors[1] : b.neighbors[0];

			for (int r = 0; r + 1 < gridSizeMax; r++) {
				bool found = closeCircle(blobs,
						chess.at<s32>(r, 0), chess.at<s32>(r, 1),
						chess.at<s32>(r + 1, 0), chess.at<s32>(r + 1, 1),
						r > 0 ? chess.at<s32>(r - 1, 0) : -1);
				if (!found) {
					break;
				}

				for (int c = 0; c + 1 < gridSizeMax; c++) {
					bool found = closeCircle(blobs,
							chess.at<s32>(r, c), chess.at<s32>(r + 1, c),
							chess.at<s32>(r, c + 1), chess.at<s32>(r + 1, c + 1),
							c > 0 ? chess.at<s32>(r, c - 1) : -1);
					if (!found) {
						break;
					}
				}
			}


			if (cv::countNonZero(chess(Range{0, 2}, Range{0, 2})) != 4) {
				continue;
			}


			{
				int i1 = chess.at<s32>(0, 0);
				int i2 = chess.at<s32>(0, 1) - foregroundBlackN;
				int i3 = chess.at<s32>(1, 1);
				int i4 = chess.at<s32>(1, 0) - foregroundBlackN;

				Vec2d p1{blackCenters.at<double>(i1, 0), blackCenters.at<double>(i1, 1)};
				Vec2d p2{whiteCenters.at<double>(i2, 0), whiteCenters.at<double>(i2, 1)};
				Vec2d p3{blackCenters.at<double>(i3, 0), blackCenters.at<double>(i3, 1)};
				Vec2d p4{whiteCenters.at<double>(i4, 0), whiteCenters.at<double>(i4, 1)};

				if (!areClockwiseOnImage(p1, p2, p3, p4)) {
					chess = chess.t();
				}
			}


			Mat chessActive = chess(Range{0, gridSize.height}, Range{0, gridSize.width});


			if (cv::countNonZero(chessActive) != gridSize.width * gridSize.height) {
				if (gridSize.height == gridSize.width) {
					continue;
				}

				if (gridSize.height % 2 == 0 && gridSize.width % 2 == 0) {
					continue;
				}

				chessActive = chess(Range{0, gridSize.width}, Range{0, gridSize.height});

				if (cv::countNonZero(chessActive) != gridSize.width * gridSize.height) {
					continue;
				}

				int currentHeight = gridSize.width;
				int currentWidth = gridSize.height;

				if (currentHeight & 1) {
					// rotate CW
					chessActive = chessActive.t();
					int flipAroundY = 1;
					cv::flip(chessActive, chessActive, flipAroundY);
				}
				else if (currentWidth & 1) {
					// rotate CCW
					chessActive = chessActive.t();
					int flipAroundX = 0;
					cv::flip(chessActive, chessActive, flipAroundX);
				}
				else {
					CV_Assert(0);
				}

				if (cv::countNonZero(chessActive) != gridSize.width * gridSize.height) {
					continue;
				}
			}

			// TODO: Check that the chessboard is EXACTLY in the correct format

			vector<Point2f> points;

			// XXX
			for (int r = 0; r < gridSize.height; r++) {
				for (int c = 0; c < gridSize.width; c++) {
					int p = chessActive.at<s32>(r, c);
					Point2f pt;
					if (p < blackN) {
						pt = Point2f{(float)blackCenters.at<double>(p, 0), (float)blackCenters.at<double>(p, 1)};
					}
					else {
						p -= foregroundBlackN;
						pt = Point2f{(float)whiteCenters.at<double>(p, 0), (float)whiteCenters.at<double>(p, 1)};
					}

					points.push_back(pt);
				}
			}

			cv::drawChessboardCorners(frame, gridSize, points, true);

			break;
		}

		printTime("Find chessboard");

		//
		// Show result and finish iteration
		//

		cv::imshow("Video feed", binary);

		showFrameRateInTitle("Video feed");
		pressedKey = cv::waitKey(1);

		printTime("Show and finish");
		std::cout << "===============================\n";
	}

	return 0;
}
