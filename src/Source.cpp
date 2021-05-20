#include "OCR.h"
#include "ResourceMatcher.h"
#include "Utils.h"

#include <chrono>
#include <opencv2/opencv.hpp>

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

void printResult(const OCR &res, const Settings &settings) {
	if (!settings.resultDir.empty()) {
		char path[256]{0,};
		snprintf(path, sizeof(path), "%s/frame-%d.jpeg", settings.resultDir.c_str(), res.frameIndex);
		printf("Matches for frame [%d] at \"%s\" {\n", res.frameIndex, path);
	} else {
		printf("Matches for frame [%d] {\n", res.frameIndex);
	}

	printf("Matches for frame [%d] {\n", res.frameIndex);

	for (int c = 0; c < int(res.matchIndices.size()); c++) {
		const ResourceMatcher &matcher = res.matchers[res.matchIndices[c]];
		if (!matcher.isMatchFound()) {
			continue;
		}

		if (!matcher.displayName.empty()) {
			printf("\t%s: ", matcher.displayName.c_str());
		} else {
			printf("\t#%d: ", c);
		}

		for (int r = 0; r < matcher.matches.size(); r++) {
			printf("(%s)", matcher.matches[r].actual.c_str());
			if (r + 1 != matcher.matches.size()) {
				printf(" ");
			}
		}
		puts("");
	}
	puts("}");
}

void printResults(const ThreadedOCR &context, VideoFile &video, const Settings &settings) {
	const OCR &first = context.results.front();
	const int matchMs = video.frameToMs(first.frameIndex);
	printf("First match found in [%s] at time %s, frame %d\n", settings.videoPath.c_str(), timeToString(matchMs).c_str(), first.frameIndex);
	if (!settings.silent) {
		for (const OCR &res : context.results) {
			printResult(res, settings);
		}
	}

	if (settings.showFrame) {
		cv::imshow("Match", first.frame);
		printf("Press any key to exit\n");
		cv::waitKey(0);
	}
}

int main(int argc, char *argv[]) {
	const Settings settings = Settings::getSettings(argc, argv);
	if (!settings.checkAndPrint()) {
		return 0;
	}
	assert(settings.isValid());
	if (!settings.silent) {
		//settings.printValues();
	}

	using namespace std;
	using namespace chrono;

	const high_resolution_clock::time_point start = high_resolution_clock::now();
	VideoFile video;
	if (!video.init(settings)) {
		printf("Failed to open file %s\n", settings.videoPath.c_str());
		return 0;
	}

	MatcherFactory matcherFactory{settings.termsFile};
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

	if (threadedOCR.foundAnyMatches()) {
		printResults(threadedOCR, video, settings);
	}

	cv::destroyAllWindows();

	return threadedOCR.remainingMatches == 0;
}
