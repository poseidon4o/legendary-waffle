#include <fstream>
#include <iostream>
#include <string>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>

#include "ResourceMatcher.h"

using namespace cv;

struct ExpBackoff {
	int getSkipFrames() const {
		return current;
	}

	void sleep() {
		current *= 2;
		current = std::min(current, max);
	}

	void mark() {
		current = min;
	}
	
	const int min = 10;
	const int max = 24 * 5; // 5 sec

	int current = min;
};

struct Settings {
	std::string filePath;
	bool showFrame = true;
	bool valid = false;

	static void printHelp() {
		std::cout << "-video\t [path]\t Path to video file to analyze\n";
		std::cout << "-show\t [1/0]\t Show frame where first detection is found\n";
	}

	static Settings getSettings(int argc, char *argv[]) {
		Settings sts;
		for (int c = 1; c < argc; c++) {
			if (argv[c] == std::string("-show") && c + 1 < argc) {
				sts.showFrame = std::string(argv[c + 1]) == "1";
				c++;
			} else if (argv[c] == std::string("-video") && c + 1 < argc) {
				sts.filePath = std::string(argv[c + 1]);
				c++;
				sts.valid = true;
			}
		}
		return sts;
	}
};


int main(int argc, char *argv[]) {
	const Settings settings = Settings::getSettings(argc, argv);
	if (!settings.valid) {
		return 1;
	}

	// Create a VideoCapture object and open the input file
	// If the input is the web camera, pass 0 instead of the video file name
	VideoCapture cap(settings.filePath);

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << std::endl;
		return -1;
	}
	enum KeyCodes {
		Enter = 13,
		Esc = 27,
		Space = 32,
	};
	tesseract::TessBaseAPI tesseract;
	tesseract.Init(TESSDATA_DIR, "eng", tesseract::OEM_TESSERACT_LSTM_COMBINED);
	tesseract.SetPageSegMode(tesseract::PSM_AUTO_OSD);
	tesseract.SetVariable("debug_file", "tesseract.log");

	int frameIdx = 0;
	ExpBackoff backoff;
	while (true) {
		Mat frame;
		cap >> frame;
		frameIdx++;

		for (int c = 0; c < backoff.getSkipFrames(); c++) {
			cap >> frame;
			frameIdx++;
		}
		if (frame.empty()) {
			std::cout << "Empty frame, stopping\n";
			break;
		}

		std::cout << "Frame " << frameIdx << std::endl;

		std::cout << "Starting match... " << std::flush;
		tesseract.SetImage(frame.data, frame.cols, frame.rows, 3, frame.step);

		tesseract.Recognize(nullptr);
		tesseract::ResultIterator* iter = tesseract.GetIterator();
		if (iter) {
			const Scalar red = {0, 0, 255};
			SpecificMatcher specMatcher;
			iter->Begin();
			do {
				if (iter->Empty(tesseract::RIL_PARA)) {
					continue;
				}
				CharPtr text(iter->GetUTF8Text(tesseract::RIL_PARA));

				int left, top, right, bottom, padding;
				if (iter->BoundingBox(tesseract::RIL_PARA, &left, &top, &right, &bottom)) {
					const Rect bbox = {{left, top}, cv::Size{right - left, bottom - top}};
					cv::rectangle(frame, bbox, {255, 0, 0});
					specMatcher.addBlock(text, bbox);
				}
			} while (iter->Next(tesseract::RIL_PARA));

			std::cout << "Match confidence " << specMatcher.getMatchConfidence() << " Strings:\n";
			if (specMatcher.getMatchConfidence() < 0.1) {
				backoff.sleep();
			} else {
				backoff.mark();
			}
			if (specMatcher.getMatchConfidence() < 0.3) {
				continue;
			}
			for (const auto &match : specMatcher.matches) {
				std::cout << match.value << "\n";
				cv::rectangle(frame, match.bbox, red);
			}
		}

		if (settings.showFrame) {
			imshow("Frame", frame);
		}
		std::cout << "done\nPress any key to exit";

		waitKey(0);
		break;
	}

	cap.release();
	destroyAllWindows();

	return 0;
}
