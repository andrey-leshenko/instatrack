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

void leshenkoThreshold(InputArray _src, OutputArray _dst, int minDiff = 10)
{
	Mat src = _src.getMat();

	CV_Assert(src.type() == CV_8U);

	Size blurKernel{3, 3};

	int minSize = std::min(src.rows, src.cols);
	int downscalingLevels = static_cast<int>(std::log2(minSize));

	Mat foundColors = Mat::zeros(src.rows, src.cols, CV_8U);

	Mat mipmap = src.clone();
	int level = 0;

	for (int i = 0; i < 3; i++) {
		cv::pyrDown(mipmap, mipmap);
		cv::blur(mipmap, mipmap, blurKernel);
		level++;
	}

	while (level <= downscalingLevels) {
		for (int r = 0; r < src.rows; r++) {
			for (int c = 0; c < src.cols; c++) {
				u8 &curr = foundColors.at<u8>(r, c);
				if (curr == 0) {
					u8 original = src.at<u8>(r, c);
					u8 thisMip = mipmap.at<u8>(r >> level, c >> level);
					if (original >= thisMip + minDiff) {
						curr = 255;
					}
					else if (original <= thisMip - minDiff) {
						curr = 127;
					}
				}
			}
		}

		cv::pyrDown(mipmap, mipmap);
		cv::blur(mipmap, mipmap, blurKernel);
		level++;
	}

	cv::threshold(foundColors, _dst, 128, 255, cv::THRESH_BINARY);
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

void findLeaksHorizontal(Mat &ternary, Mat &outLeakMask, int maxDistance)
{
	CV_Assert(ternary.type() == CV_8U);

	outLeakMask.create(ternary.rows, ternary.cols, ternary.type());
	outLeakMask.setTo(0);

	for (int r = 0; r < ternary.rows; r++) {
		u8 lastColor = 0;
		int distance = maxDistance + 1;

		u8 *curr = ternary.ptr<u8>(r);

		for (int c = 0; c < ternary.cols; c++) {
			if (*curr != lastColor && *curr != 127) {
				if (distance <= maxDistance) {
					int start = c - distance;

					for (int i = start; i < c; i++) {
						outLeakMask.at<u8>(r, i) = 255;
					}
				}

				lastColor = *curr;
				distance = 0;
			}

			distance++;
			curr++;
		}
	}
}

void findLeaksDiagonal(Mat &ternary, Mat &outLeakMask, int maxDistance)
{
	// This should help when the chessboard is nice and parallel
}

void findLeaks(Mat &ternary, Mat &outLeakMask, int maxDistance)
{
	CV_Assert(ternary.type() == CV_8U);

	outLeakMask.create(ternary.rows, ternary.cols, ternary.type());
	cv::bitwise_and(outLeakMask, 0, outLeakMask);

	auto findLeaksHorizontal = [](Mat &ternary, Mat &outLeakMask, int maxDistance)
	{
		for (int r = 0; r < ternary.rows; r++) {
			u8 lastColor = 0;
			int distance = maxDistance + 1;

			u8 *curr = ternary.ptr<u8>(r);

			for (int c = 0; c < ternary.cols; c++) {
				if (*curr != lastColor && *curr != 127) {
					if (distance <= maxDistance) {
						int start = c - distance;

						for (int i = start; i < c; i++) {
							outLeakMask.at<u8>(r, i) = 255;
						}
					}

					lastColor = *curr;
					distance = 0;
				}

				distance++;
				curr++;
			}
		}
	};

	findLeaksHorizontal(ternary, outLeakMask, maxDistance);
	Mat ternaryT = ternary.t();
	Mat outLeakMaskT = outLeakMask.t();
	findLeaksHorizontal(ternaryT, outLeakMaskT, maxDistance);
	outLeakMask = outLeakMaskT.t();
}

void findSquareNoise(Mat &ternary, Mat &outNoiseMask, int maxDistance)
{
	findLeaksHorizontal(ternary, outNoiseMask, maxDistance);

	Mat ternaryT = ternary.t();
	Mat verticalNoise;

	findLeaksHorizontal(ternaryT, verticalNoise, maxDistance);

	cv::bitwise_and(outNoiseMask, verticalNoise.t(), outNoiseMask);
}

void createTernary(const Mat &blackMask, const Mat &whiteMask, Mat &ternary)
{
	// NOTE(Andrey): In case of intersections, blacks win
	Mat blackMaskInv;
	cv::bitwise_not(blackMask, blackMaskInv);
	cv::bitwise_and(whiteMask, blackMaskInv, whiteMask);

	ternary.create(blackMask.rows, blackMask.cols, CV_8U);
	ternary.setTo(Scalar{127});
	cv::subtract(ternary, blackMask, ternary);
	cv::bitwise_or(ternary, whiteMask, ternary);
}


void ternerizeShrink(const Mat &binary, Mat &ternary, Mat &blackMask, Mat &whiteMask)
{
	whiteMask = binary.clone();
	cv::bitwise_not(whiteMask, blackMask);

	cv::dilate(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 2);
	cv::erode(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 6);
	cv::dilate(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 1);

	cv::dilate(blackMask, blackMask, Mat{}, Point{-1, -1}, 2);
	cv::erode(blackMask, blackMask, Mat{}, Point{-1, -1}, 6);
	cv::dilate(blackMask, blackMask, Mat{}, Point{-1, -1}, 1);

	createTernary(blackMask, whiteMask, ternary);
}

void ternerizeGrow(const Mat &binary, Mat &ternary, Mat &blackMask, Mat &whiteMask)
{
	whiteMask = binary.clone();
	cv::bitwise_not(whiteMask, blackMask);

	//cv::erode(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 2);
	cv::dilate(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 2);
	//cv::erode(blackMask, blackMask, Mat{}, Point{-1, -1}, 2);
	cv::dilate(blackMask, blackMask, Mat{}, Point{-1, -1}, 2);

	Mat notInBoth;
	cv::bitwise_xor(whiteMask, blackMask, notInBoth);
	cv::bitwise_and(whiteMask, notInBoth, whiteMask);
	cv::bitwise_and(blackMask, notInBoth, blackMask);

	cv::dilate(whiteMask, whiteMask, Mat{}, Point{-1, -1}, 1);
	cv::dilate(blackMask, blackMask, Mat{}, Point{-1, -1}, 1);

	createTernary(blackMask, whiteMask, ternary);
}

void ternerizeRemoveLeaks(const Mat &binary, Mat &ternary, Mat &blackMask, Mat &whiteMask)
{
	ternary = binary.clone();
	Mat noise;

	for (int i = 1; i <= 4; i++) {
		findSquareNoise(ternary, noise, i);
		cv::bitwise_and(ternary, 0, ternary, noise);
		cv::bitwise_or(ternary, 127, ternary, noise);
	}

	Mat leaks = noise;

	for (int i = 4; i <= 4; i++) {
		findLeaks(ternary, leaks, i);
		cv::bitwise_and(ternary, 0, ternary, leaks);
		cv::bitwise_or(ternary, 127, ternary, leaks);
	}

	cv::inRange(ternary, 0, 0, blackMask);
	cv::inRange(ternary, 255, 255, whiteMask);
}

struct SegmentationResult
{
	int blackN;
	int whiteN;
	int totalN;

	Mat labels;
	Mat colors;
};

void drawCComps(const Mat &labels, Mat &dst)
{
	CV_Assert(labels.type() == CV_32S);
	vector<cv::Vec3b> palette(32);

	cv::RNG rng(7091998);

	for (int i = 0; i < palette.size(); i++) {
		palette[i] = {(u8)rng.uniform(0, 180), (u8)rng.uniform(100, 200), (u8)rng.uniform(50, 255)};
	}

	cv::cvtColor(palette, palette, cv::COLOR_HSV2BGR);

	dst.create(labels.size(), CV_8UC3);

	auto labelIt = labels.begin<s32>();
	auto dstIt = dst.begin<cv::Vec3b>();

	auto labelEnd = labels.end<s32>();

	while (labelIt != labelEnd) {
		int comp = *labelIt;
		if (comp == 0) {
			*dstIt = {0, 0, 0};
		}
		else {
			*dstIt = palette[comp % palette.size()];
		}

		labelIt++;
		dstIt++;
	}
}

void doSegmentationTernary(const Mat &binary, SegmentationResult &seg)
{
	Mat ternary;
	Mat blackMask;
	Mat whiteMask;

	//ternerizeShrink(binary, ternary, blackMask, whiteMask);
	ternerizeGrow(binary, ternary, blackMask, whiteMask);
	//ternerizeRemoveLeaks(binary, ternary, blackMask, whiteMask);

	Mat whiteLabels;

	seg.blackN = cv::connectedComponents(blackMask, seg.labels, 4, CV_32S) - 1;
	seg.whiteN = cv::connectedComponents(whiteMask, whiteLabels, 4, CV_32S) - 1;

	cv::add(seg.labels, seg.blackN, seg.labels, whiteMask);

	seg.labels += whiteLabels;
	seg.totalN = seg.whiteN + seg.blackN + 1;
	seg.colors = ternary;
}

int segmentWhite(const Mat &whiteMask, const Mat &blackMask, Mat &outLabels)
{
	Mat dist;
	Mat markers;

	cv::distanceTransform(whiteMask, dist, cv::DIST_L1, cv::DIST_MASK_3);
	cv::threshold(dist, markers, 3, 255, CV_THRESH_BINARY_INV);

	markers.convertTo(markers, CV_8U);

	Mat segmentDist;

	cv::distanceTransform(
			markers,
			segmentDist, // TODO(Andrey): Check out cv::noArray()
			outLabels,
			cv::DIST_L1,
			cv::DIST_MASK_3,
			cv::DIST_LABEL_CCOMP);

	outLabels.setTo(0, blackMask);
}

void doSegmentationNew(const Mat &binary, SegmentationResult &seg)
{
	Mat whiteMask = binary;
	Mat blackMask;

	cv::bitwise_not(whiteMask, blackMask);

	Mat blackLabels = seg.labels;
	Mat whiteLabels;

	segmentWhite(whiteMask, blackMask, whiteLabels);
	segmentWhite(blackMask, whiteMask, blackLabels);

	double max;

	cv::minMaxIdx(blackLabels, NULL, &max);
	seg.blackN = (int)max;

	cv::minMaxIdx(whiteLabels, NULL, &max);
	seg.whiteN = (int)max;

	seg.labels = blackLabels;

	cv::add(seg.labels, seg.blackN, seg.labels, whiteMask);
	seg.labels += whiteLabels;

	seg.totalN = seg.whiteN + seg.blackN + 1;
	seg.colors = binary;
}

void findNeighborsHorizontal(vector<Blob> &blobs, Mat colors, Mat labels, int maxDistance)
{
	int imRows = labels.rows;
	int imCols = labels.cols;

	for (int r = 0; r < imRows; r++) {
		u8 lastColor = 127;
		s32 lastComp = 0;
		int distance = 0;

		for (int c = 0; c < imCols; c++) {
			u8 currColor = colors.at<u8>(r, c);
			s32 currComp = labels.at<s32>(r, c);

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

void findNeighbors(vector<Blob> &blobs, Mat colors, Mat labels, int maxDistance)
{
	findNeighborsHorizontal(blobs, colors, labels, maxDistance);
	Mat colorsT = colors.t();
	Mat labelsT = labels.t();
	findNeighborsHorizontal(blobs, colorsT, labelsT, maxDistance);
}

struct CCompCenterStats
{
	long sumX;
	long sumY;
	long count;
};

void findCCompCenters(SegmentationResult &seg, vector<Point2f> &centers)
{
	vector<CCompCenterStats> stats(seg.totalN, {0});

	Mat &labels = seg.labels;

	int imRows = labels.rows;
	int imCols = labels.cols;

	for (int r = 0; r < imRows; r++) {
		s32 *p = labels.ptr<s32>(r);

		for (int c = 0; c < imCols; c++) {
			auto &s = stats[*p];
			s.sumX += c;
			s.sumY += r;
			s.count++;

			p++;
		}
	}

	centers.resize(seg.totalN);

	for (int i = 0; i < seg.totalN; i++) {
		auto &s = stats[i];
		if (s.count == 0) {
			centers[i] = {0, 0};
		}
		else {
			centers[i] = {(float)(s.sumX / s.count), (float)(s.sumY / s.count)};
		}
	}
}

void drawNeighborGraph(Mat &dst, const SegmentationResult &seg, const vector<Blob> &blobs, const vector<Point2f> &centers)
{
	for (int i = 1; i <= seg.whiteN; i++) {
		cv::circle(dst, centers[i], 2, Scalar{180, 10, 10}, -1);
	}

	for (int i = seg.whiteN + 1; i < seg.totalN; i++) {
		cv::circle(dst, centers[i], 2, Scalar{10, 10, 180}, -1);
	}

	for (int i = 1; i < seg.totalN; i++) {
		Blob b = blobs[i];
		Point2f p1 = centers[i];

		for (int k = 0; k < 4; k++) {
			int n = b.neighbors[k];

			if (n == 0) {
				break;
			}

			Point2f p2 = centers[n];

			cv::line(dst, p1, p2, Scalar{10, 10, 10});
		}
	}
}

bool findChessboardInGraph(SegmentationResult &seg, vector<Blob> &blobs, vector<Point2f> &centers, Size chessSize, vector<Point2f> &chessPoints)
{
	Mat chess;

	int chessSizeMax = std::max(chessSize.width, chessSize.height);
	chess.create(chessSizeMax, chessSizeMax, CV_32S);
	
	// NOTE(Andrey): So cv::drawChessboardCorners won't draw last frame's points
	chessPoints.resize(0);

	CV_Assert(chessSize.height > 1 && chessSize.width > 1);

	for (int i = 1; i < seg.blackN; i++) {
		Blob &b = blobs[i];

		if (b.neighbors[3] || !b.neighbors[2]) {
			continue;
		}

		chess.setTo(0);
		chess.at<s32>(0, 0) = i;
		chess.at<s32>(0, 1) = blobs[b.neighbors[0]].invalid ?
			b.neighbors[1] : b.neighbors[0];

		for (int r = 0; r + 1 < chessSizeMax; r++) {
			bool found = closeCircle(blobs,
					chess.at<s32>(r, 0), chess.at<s32>(r, 1),
					chess.at<s32>(r + 1, 0), chess.at<s32>(r + 1, 1),
					r > 0 ? chess.at<s32>(r - 1, 0) : -1);
			if (!found) {
				break;
			}

			for (int c = 0; c + 1 < chessSizeMax; c++) {
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
			int i2 = chess.at<s32>(0, 1);
			int i3 = chess.at<s32>(1, 1);
			int i4 = chess.at<s32>(1, 0);

			using cv::Vec2f;

			Vec2f p1 = (Vec2f)centers[chess.at<s32>(0, 0)];
			Vec2f p2 = (Vec2f)centers[chess.at<s32>(0, 1)];
			Vec2f p3 = (Vec2f)centers[chess.at<s32>(1, 1)];
			Vec2f p4 = (Vec2f)centers[chess.at<s32>(1, 0)];

			if (!areClockwiseOnImage(p1, p2, p3, p4)) {
				chess = chess.t();
			}
		}

		Mat chessActive = chess(Range{0, chessSize.height}, Range{0, chessSize.width});


		if (cv::countNonZero(chessActive) != chessSize.width * chessSize.height) {
			if (chessSize.height == chessSize.width) {
				continue;
			}

			if (chessSize.height % 2 == 0 && chessSize.width % 2 == 0) {
				continue;
			}

			chessActive = chess(Range{0, chessSize.width}, Range{0, chessSize.height});

			if (cv::countNonZero(chessActive) != chessSize.width * chessSize.height) {
				continue;
			}

			int currentHeight = chessSize.width;
			int currentWidth = chessSize.height;

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

			if (cv::countNonZero(chessActive) != chessSize.width * chessSize.height) {
				continue;
			}
		}

		// TODO: Check that the chessboard is EXACTLY in the correct format

		chessPoints.resize(0);

		for (int r = 0; r < chessSize.height; r++) {
			for (int c = 0; c < chessSize.width; c++) {
				int p = chessActive.at<s32>(r, c);
				chessPoints.push_back(centers[p]);
			}
		}

		return true;
	}

	return false;
}


int main(int argc, char* argv[])
{
	int cameraIndex = 0;
	Size chessSize{5, 4};

	VideoCapture cap;
	Mat loadedImage;

	Mat frame;
	Mat grayscale;
	Mat binary;
	Mat result;

	SegmentationResult seg;
	vector<Point2f> centers;
	vector<Blob> blobs;
	vector<Point2f> chessPoints;
	bool chessboardFound;

	int mode = 6;
	bool paused = false;
	bool newThreshold = false;
	bool oldTernerization = true;
	bool newSegmentation = true;
	bool failTrigger = false;

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

	if (cameraIndex >= 0) {
		cap.grab();
	}

	//cv::namedWindow("Video feed", cv::WINDOW_NORMAL);

	int pressedKey = 0;

	while (pressedKey != 'q') {
		printTime("Start");

		//
		// Capture a frame
		//

		if (!paused) {
			if (cameraIndex >= 0) {
				cap.retrieve(frame);
				cap.grab();
			}
			else {
				frame = loadedImage.clone();
			}
		}

		printTime("Capture image");

		//
		// Threshold
		//

		cv::cvtColor(frame, grayscale, CV_BGR2GRAY);

		if (!newThreshold) {
			cv::threshold(grayscale, binary, 80, 255, CV_THRESH_BINARY);
		}
		else {
			leshenkoThreshold(grayscale, binary);
		}

		printTime("Threshold");

		//
		// Segmentation
		//

		if (!newSegmentation) {
			doSegmentationTernary(binary, seg);
		}
		else {
			doSegmentationNew(binary, seg);
		}

		findCCompCenters(seg, centers);

		printTime("Segmentation");

		//
		// Find neighbors
		//

		{
			// NOTE(Andrey): Will never decrease capacity
			blobs.resize(seg.totalN);
			Blob empty = {0};
			std::fill(blobs.begin(), blobs.end(), empty);
		}

		findNeighbors(blobs, seg.colors, seg.labels, 10);

		printTime("Find neighbors");

		//
		// Find chessboard in graph
		//

		chessboardFound = findChessboardInGraph(seg, blobs, centers, chessSize, chessPoints);

		printTime("Find chessboard");

		//
		// Show results and finish iteration
		//
		//

		if (mode == 1) {
			cv::imshow("Video feed", frame);
		}
		else if (mode == 2) {
			cv::imshow("Video feed", grayscale);
		}
		else if (mode == 3) {
			cv::imshow("Video feed", binary);
		}
		else if (mode == 4 || mode == 5) {
			Mat comps;
			drawCComps(seg.labels, comps);

			if (mode == 5) {
				drawNeighborGraph(comps, seg, blobs, centers);
			}

			cv::imshow("Video feed", comps);
		}
		else if (mode >= 6) {
			result = frame.clone();
			cv::drawChessboardCorners(result, chessSize, chessPoints, chessboardFound);

			cv::imshow("Video feed", result);
		}

		showFrameRateInTitle("Video feed");
		pressedKey = cv::waitKey(1);

		if (pressedKey >= '0' && pressedKey <= '9') {
			mode = pressedKey - '0';
		}
		else if (pressedKey == ' ') {
			paused = !paused;
		}
		else if (pressedKey == 't') {
			newThreshold = !newThreshold;
		}
		else if (pressedKey == 'o') {
			oldTernerization = !oldTernerization;
		}
		else if (pressedKey == 's') {
			newSegmentation = !newSegmentation;
		}
		else if (pressedKey == 'c') {
			failTrigger = true;
		}

		if (failTrigger && !chessboardFound) {
			paused = true;
			failTrigger = false;
		}

		printTime("Show and finish\n=====================");
	}

	return 0;
}
