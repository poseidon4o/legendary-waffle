#include "OCR.h"
#include "RuleMatcher.h"
#include "Utils.h"

#include <chrono>
#include <opencv2/opencv.hpp>

#include <map>

#pragma optimize("", off)


void printHardMatch(const MatchResult &res, VideoFile &video, const Settings &settings) {
	const std::string &frameTime = timeToString(video.frameToMs(res.frameIndex));
	if (!settings.resultDir.empty()) {
		char path[256]{0,};
		snprintf(path, sizeof(path), "%s/frame-%s.jpeg", settings.resultDir.c_str(), frameTime.c_str());
		printf("Matches for frame [%d] (%s) at \"%s\" {\n", res.frameIndex, frameTime.c_str(), path);
	} else {
		printf("Matches for frame [%d] (%s) {\n", res.frameIndex, frameTime.c_str());
	}

	const MatcherList &whitelist = res.ruleSet.getWhitelist();
	for (int c = 0; c < int(res.whitelistIndices.size()); c++) {
		const RuleMatcher &matcher = whitelist[res.whitelistIndices[c]];
		if (!matcher.isMatchFound() || matcher.descriptor().isSoftMatch) {
			continue;
		}

		printf("\t%s: ", matcher.descriptor().name.c_str());

		const RuleMatch& termList = matcher.getMatchedTerms();
		for (int r = 0; r < termList.size(); r++) {
			printf("(%s)", termList[r].actual.c_str());
			if (r + 1 != termList.size()) {
				printf(" ");
			}
		}
		puts("");
	}
	puts("}");
}

void printResults(const ThreadedOCR &context, VideoFile &video, const Settings &settings, const MatcherFactory &matcherFactory) {
	const MatchResult &first = context.results.front();
	const ms matchMs = video.frameToMs(first.frameIndex);
	printf("First match found in [%s] at time %s, frame %d\n", settings.videoPath.c_str(), timeToString(matchMs).c_str(), first.frameIndex);

	if (!settings.silent) {
		struct SoftMatchInfo {
			std::vector<int> frames;
		};
		std::map<int, SoftMatchInfo> softMatches;
		for (const MatchResult &res : context.results) {
			if (res.matchType & MatchResult::HardMatch) {
				printHardMatch(res, video, settings);
			}
			if (res.matchType & MatchResult::SoftMatch) {
				for (int c = 0; c < int(res.whitelistIndices.size()); c++) {
					SoftMatchInfo &info = softMatches[res.whitelistIndices[c]];
					info.frames.push_back(res.frameIndex);
				}
			}
		}

		RuleSet set;
		matcherFactory.create(set);
		// need only names
		const MatcherList &whitelist = set.getWhitelist();
		for (const auto &pair : softMatches) {
			printf("Soft match [%s] at {", whitelist[pair.first].descriptor().name.c_str());
			for (int c = 0; c < int(pair.second.frames.size()); c++) {
				const std::string frameStr = timeToString(video.frameToMs(pair.second.frames[c]));
				printf("[%d %s]", pair.second.frames[c], frameStr.c_str());
			}
			puts("}");
		}
	}

	if (settings.showFrame) {
		cv::imshow("TermMatch", first.frame);
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
	const ms processingMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	if (!settings.silent) {
		printf("Processing time time %s [%dms]\n", timeToString(processingMs).c_str(), int(processingMs.count()));
	}

	if (threadedOCR.foundAnyMatches()) {
		printResults(threadedOCR, video, settings, matcherFactory);
	}

	cv::destroyAllWindows();

	return threadedOCR.remainingMatches == 0;
}
