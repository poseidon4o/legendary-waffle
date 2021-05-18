#pragma once

#if _WIN32
#define assert(test) do { ( (test) ? (void)0 : __debugbreak() )} while(0)
#else
#include <cassert>
#endif

#include <algorithm>
#include <mutex>
#include <string>

typedef std::lock_guard<std::mutex> lock_guard;
typedef std::unique_lock<std::mutex> unique_lock;

struct Settings {
	std::string videoPath;
	std::string matchersFile;
	bool showFrame = true;
	int threadCount = -1;
	int frameSkip = 24;
	bool silent = false;
	bool doCrop = false;

	bool isValid() const {
		return !videoPath.empty() && !matchersFile.empty();
	}

	void printValues() const {
		printf("-videoPath     [path]     (%s) Path to video file to analyze\n", videoPath.c_str());
		printf("-matchersFile  [path]     (%s) Path to matches file containing search strings\n", matchersFile.c_str());
		printf("-show          [1/0]      (%d) Show frame where first detection is found\n", threadCount);
		printf("-silent        [1/0]      (%d) Print only on error and match found\n", silent);
		printf("-doCrop        [1/0]      (%d) Crop image to upper/left 1/4th\n", doCrop);
		printf("-threadCount   [<number>] (%d) Number of threads\n", threadCount);
		printf("-frameSkip     [<number>] (%d) Number of threads\n", frameSkip);
		printf("-help          []         (  )   Show this help\n");
	}

	static void printHelp() {
		const Settings defaults;
		defaults.printValues();
	}

	static Settings getSettings(int argc, char *argv[]) {
		Settings sts;
		for (int c = 1; c < argc; c++) {
			if (argv[c] == std::string("-help")) {
				return sts;
			} else if (argv[c] == std::string("-show") && c + 1 < argc) {
				sts.showFrame = std::string(argv[c + 1]) == "1";
				c++;
			} else if (argv[c] == std::string("-silent") && c + 1 < argc) {
				sts.silent = std::string(argv[c + 1]) == "1";
				c++;
			} else if (argv[c] == std::string("-doCrop") && c + 1 < argc) {
				sts.doCrop = std::string(argv[c + 1]) == "1";
				c++;
			} else if (argv[c] == std::string("-video") && c + 1 < argc) {
				sts.videoPath = std::string(argv[c + 1]);
				c++;
			} else if (argv[c] == std::string("-threadCount") && c + 1 < argc) {
				sts.threadCount = atoi(argv[c + 1]);
				sts.threadCount = std::max(-1, std::min<int>(std::thread::hardware_concurrency(), sts.threadCount));
				c++;
			} else if (argv[c] == std::string("-frameSkip") && c + 1 < argc) {
				sts.frameSkip = atoi(argv[c + 1]);
				sts.frameSkip = std::max(1, sts.frameSkip);
				c++;
			} else if (argv[c] == std::string("-matchersFile") && c + 1 < argc) {
				sts.matchersFile = std::string(argv[c + 1]);
				c++;
			}
		}
		return sts;
	}
};