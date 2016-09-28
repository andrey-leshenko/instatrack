#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

using std::vector;

void connectedComponents(const u8 *img, int rows, int cols)
{
	int size = rows * cols;
	s32 currColor = -1;
	s32 last = !img[0];
	vector<int> eq;

	s32 *comp = new s32[rows * cols];

	for (int i = 0; i < cols; i++) {
		if (img[i] != last) {
			currColor++;
			eq.push_back(currColor);
			last = img[i];
		}
		comp[i] = currColor;
	}

	for (int r = 1; r < rows; r++) {
		last = !img[r * cols];
		for (int c = 0; c < cols; c++) {
			int i = r * cols + c;

			if (img[i] != last) {
				currColor++;
				eq.push_back(currColor);
				last = img[i];
			}

			comp[i] = currColor;
		}
	}

	for (int i = cols; i < rows * cols; i++) {
		int top = i - cols;

		int compI = comp[i];
		int compTop = comp[top];

		if (img[top] == img[i] && eq[compTop] < eq[compI]) {
			eq[compI] = eq[compTop];
		}
	}

	for (int i = 0; i < (rows - 1) * cols; i++) {
		int bot = i + cols;

		int compI = comp[i];
		int compBot = comp[bot];

		if (img[bot] == img[i] && eq[compBot] < eq[compI]) {
			eq[compI] = eq[compBot];
		}
	}

	/*
	for (int i = 1; r < rows; r++) {
		last = !img[r * cols];
		for (int c = 0; c < cols; c++) {
			int i = r * cols + c;

			if (img[i] != last) {
				currColor++;
				eq.push_back(currColor);
				last = img[i];
			}

			comp[i] = currColor;

			int top = i - cols;

			if (img[top] == img[i] && eq[comp[top]] < eq[currColor]) {
				eq[currColor] = eq[comp[top]];
			}
		}
	}
	*/

	for (int i = 0; i < rows * cols; i++) {
		comp[i] = eq[comp[i]];
	}
	
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			printf("%2d", comp[r * cols + c]);
		}
		printf("\n");
	}
}

const char *bitmapStr =
"XXXXXXXXXXXXXXXXXXXX"
"XX  XXX     XX  XX X"
"XXX     XXXXXX    XX"
"X   XXXXX  XXXXX XXX"
"XX   XXXXX  XX  XX X"
"XX  XX  XXX    XX  X"
"XX  XXX   XXXXXXXXXX"
"XXXXXXXXXXXXXXXXXXXX";

int main(int argc, char* argv[])
{
	int rows = 8;
	int cols = 20;

	u8 *bitmap = new u8[rows * cols];
	memcpy(bitmap, bitmapStr, rows * cols);

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			printf("%c", bitmap[r * cols + c]);
		}
		printf("\n");
	}

	printf("\n\n");

	connectedComponents(bitmap, rows, cols);

	return 0;
}
