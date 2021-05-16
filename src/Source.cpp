#include "OCR.h"
#include "ResourceMatcher.h"
#include "Utils.hpp"

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


int main(int argc, char *argv[]) {
	const Settings settings = Settings::getSettings(argc, argv);
	if (!settings.valid) {
		Settings::printHelp();
		return 1;
	}

	VideoFile video;
	if (!video.init(settings)) {
		printf("Failed to open file %s\n", settings.filePath.c_str());
		return 2;
	}

	ThreadedOCR threadedOCR(video);
	if (!threadedOCR.start(settings.threadCount)) {
		puts("Failed to start threads");
		return 4;
	}

	threadedOCR.waitFinish();

	if (threadedOCR.result.matchFound()) {
		cv::imshow("Match", threadedOCR.result.frame);
		printf("Press any key to exit\n");
		cv::waitKey(0);
	}

	enum KeyCodes {
		Enter = 13,
		Esc = 27,
		Space = 32,
	};

	cv::destroyAllWindows();

	return 0;
}
