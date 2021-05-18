#include "OCR.h"
#include "ResourceMatcher.h"
#include "Utils.hpp"

int main(int argc, char *argv[]) {
	const Settings settings = Settings::getSettings(argc, argv);
	if (!settings.isValid()) {
		Settings::printHelp();
		return 1;
	}

	VideoFile video;
	if (!video.init(settings)) {
		printf("Failed to open file %s\n", settings.videoPath.c_str());
		return 2;
	}

	MatcherFactory matcherFactory{settings.matchersFile};
	if (!matcherFactory.init()) {
		puts("Failed to load matchers");
		return 3;
	}
	matcherFactory.showInfo();

	ThreadedOCR threadedOCR(settings, matcherFactory, video);
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
