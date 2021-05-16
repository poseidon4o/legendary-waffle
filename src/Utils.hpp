#pragma once

#if _WIN32
#define assert(test) do { ( (test) ? (void)0 : __debugbreak() )} while(0)
#else
#include <cassert>
#endif

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

	bool isValid() const {
		return !videoPath.empty() && !matchersFile.empty();
	}

	static void printHelp() {
		Settings defaults;
		printf("-videoPath\t [path]\t (%s) Path to video file to analyze\n", defaults.videoPath.c_str());
		printf("-matchersFile\t [path]\t (%s) Path to video file to analyze\n", defaults.matchersFile.c_str());
		printf("-show\t\t [1/0]\t (%d) Show frame where first detection is found\n", defaults.threadCount);
		printf("-threadCount\t\t [<number>]\t (%d) Number of threads\n", defaults.threadCount);
		printf("-frameSkip\t\t [<number>]\t (%d) Number of threads\n", defaults.frameSkip);
		printf("-help\t\t []\t () Show this help\n");
	}

	static Settings getSettings(int argc, char *argv[]) {
		Settings sts;
		for (int c = 1; c < argc; c++) {
			if (argv[c] == std::string("-help")) {
				return sts;
			} else if (argv[c] == std::string("-show") && c + 1 < argc) {
				sts.showFrame = std::string(argv[c + 1]) == "1";
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