#include "../src/base.hpp"

void findSaddlePoints(InputArray _src, OutputArray _dst, int _regionSize)
{
	static Mat tmp;

	Mat src = _src.getMat();
	CV_Assert(src.depth() == CV_8U);
	CV_Assert(src.isContinuous());

	_dst.create(src.size(), CV_8U);
	Mat dst = _dst.getMat();

	tmp.create(src.size(), CV_16UC2);

	u8 *srcData = src.ptr<u8>();
	u16 *tmpData = tmp.ptr<u16>();
	u8 *dstData = dst.ptr<u8>();


	int srcRows = src.rows;
	int srcCols = src.cols;

	std::memset(tmpData, 0, srcRows * srcCols * sizeof(u16) * tmp.channels());
	std::memset(dstData, 0, srcRows * srcCols * sizeof(u8));

	for (int r = 1; r < srcRows; r++) {
		for (int c = 1; c < srcCols; c++) {
			int i = r * srcCols + c;
			int i2 = i * 2;

			tmpData[i2] = tmpData[i2 - 1 * 2];
			tmpData[i2 + 1] = tmpData[i2 + 1 - srcCols * 2];

			if (srcData[i] != srcData[i - 1]) {
				tmpData[i2]++;
			}
			if (srcData[i] != srcData[i - srcCols]) {
				tmpData[i2 + 1]++;
			}
		}
	}

	for (int r = 0; r < srcRows; r++) {
		for (int c = 0; c < srcCols; c++) {
			for (int regionSize = 1; regionSize <= _regionSize; regionSize++) {
			int top = (r >= regionSize) ? r - regionSize : 0;
			int bot = (r + regionSize < srcRows) ? r + regionSize : srcRows - 1;
			int left = (c >= regionSize) ? c - regionSize : 0;
			int right = (c + regionSize < srcCols) ? c + regionSize : srcCols - 1;

			int nw = top * srcCols + left;
			int ne = top * srcCols + right;
			int sw = bot * srcCols + left;
			int se = bot * srcCols + right;

			nw <<= 1;
			ne <<= 1;
			se <<= 1;
			sw <<= 1;

			int switches1 = tmpData[ne] - tmpData[nw];
			int switches2 = tmpData[se + 1] - tmpData[ne + 1];
			int switches3 = tmpData[se] - tmpData[sw];
			int switches4 = tmpData[sw + 1] - tmpData[nw + 1];

			int switches = switches1 + switches2 + switches3 + switches4;

			if (switches >= 4 && ((switches1 && switches3) || (switches2 && switches4))) {
				int i = r * srcCols + c;
				dstData[i] = 127;
			}
			if (switches >= 4) {
				int i = r * srcCols + c;
				//dstData[i] = 127;
			}
			//dstData[r * srcCols + c] = tmpData[(r * srcCols + c) * 2];
			//dstData[r * srcCols + c] = tmpData[(r * srcCols + c) * 2 + 1];
			//dstData[r * srcCols + c] = switches;
			}
		}
	}
}

int main(int argc, char* argv[])
{
	int cameraIndex = 1;

	VideoCapture cap{cameraIndex};

	if (!cap.isOpened()) {
		std::cerr << "Error: Couldn't capture camera number " << cameraIndex << '\n';
		return -1;
	}

	cap.set(cv::CAP_PROP_FPS, 60);

	int imRows;
	int imCols;

	{
		Mat tmp;
		cap >> tmp;

		imCols = tmp.cols;
		imRows = tmp.rows;
	}

	Mat frame;
	Mat grayscale;
	Mat binary;
	Mat saddle;

	int pressedKey = 0;

	int mode = 3;
	int regionSize = 1;

	while (pressedKey != 'q') {
		cap >> frame;

		cv::cvtColor(frame, grayscale, CV_BGR2GRAY);
		cv::threshold(grayscale, binary, 80, 200, CV_THRESH_BINARY);
		findSaddlePoints(binary, saddle, regionSize);
		cv::add(binary, saddle, binary);
		//binary = saddle.clone();

		if (mode == 1) {
			cv::imshow("w", grayscale);
		}
		else if (mode == 2) {
			cv::imshow("w", saddle);
		}
		else if (mode >= 3) {
			Mat combined;
			cv::addWeighted(binary, 0.5, saddle, 1, 0, combined);
		}

		showFrameRateInTitle("w");
		pressedKey = cv::waitKey(1);

		if (pressedKey >= '0' && pressedKey <= '9') {
			mode = pressedKey - '0';
		}

		switch (pressedKey) {
			case 'k':
				regionSize++;
				break;
			case 'j':
				if (regionSize) {
					regionSize--;
				}
				break;
		}
	}

	return 0;
}
