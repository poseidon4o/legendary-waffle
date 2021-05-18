#include "OCR.h"
#include "ResourceMatcher.h"
#include "Utils.hpp"

#include <chrono>

std::string timeToString(int msCount) {
	using namespace std;
	using namespace chrono;

	const milliseconds ms(msCount);
	char buff[64];
	snprintf(buff, sizeof(buff), "%d:%d:%d",
		int(duration_cast<hours>(ms).count()),
		int(duration_cast<minutes>(ms).count()),
		int(duration_cast<seconds>(ms).count())
	);
	return buff;
}

int main(int argc, char *argv[]) {
	const Settings settings = Settings::getSettings(argc, argv);
	if (!settings.isValid()) {
		Settings::printHelp();
		return 0;
	}

	if (!settings.silent) {
		settings.printValues();
	}

	using namespace std;
	using namespace chrono;

	const high_resolution_clock::time_point start = high_resolution_clock::now();
	VideoFile video;
	if (!video.init(settings)) {
		printf("Failed to open file %s\n", settings.videoPath.c_str());
		return 0;
	}

	MatcherFactory matcherFactory{settings.matchersFile};
	if (!matcherFactory.init()) {
		puts("Failed to load matchers");
		return 0;
	}
	matcherFactory.showInfo();

	ThreadedOCR threadedOCR(settings, matcherFactory, video);
	if (!threadedOCR.start(settings.threadCount)) {
		puts("Failed to start threads");
		return 0;
	}

	threadedOCR.waitFinish();

	const high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	const int processingMs = int(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
	if (!settings.silent) {
		printf("Processing time time %s [%dms]\n", timeToString(processingMs).c_str(), processingMs);
	}

	if (threadedOCR.result.matchFound()) {
		const int matchMs = video.frameToMs(threadedOCR.result.frameIndex);
		printf("Match found at time %s, frame %d\n", timeToString(matchMs).c_str(), threadedOCR.result.frameIndex);
		if (settings.showFrame) {
			cv::imshow("Match", threadedOCR.result.frame);
			printf("Press any key to exit\n");
			cv::waitKey(0);
		}
	}

	cv::destroyAllWindows();

	return threadedOCR.result.matchFound();
}
