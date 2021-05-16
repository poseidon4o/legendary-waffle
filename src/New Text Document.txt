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
	std::string filePath;
	bool showFrame = true;
	bool valid = false;
	int threadCount = -1;

	static void printHelp() {
		Settings defaults;
		printf("-video\t [path]\t (%s) Path to video file to analyze\n", defaults.filePath.c_str());
		printf("-show\t [1/0]\t (%d) Show frame where first detection is found\n", defaults.threadCount);
		printf("-threadCount\t [<number>]\t (%d) Number of threads\n", defaults.threadCount);
		printf("-help\t []\t () Show this help\n");
	}

	static Settings getSettings(int argc, char *argv[]) {
		Settings sts;
		for (int c = 1; c < argc; c++) {
			if (argv[c] == std::string("-help")) {
				sts.valid = false;
				return sts;
			} else if (argv[c] == std::string("-show") && c + 1 < argc) {
				sts.showFrame = std::string(argv[c + 1]) == "1";
				c++;
			} else if (argv[c] == std::string("-video") && c + 1 < argc) {
				sts.filePath = std::string(argv[c + 1]);
				c++;
				sts.valid = true;
			} else if (argv[c] == std::string("-threadCount") && c + 1 < argc) {
				sts.threadCount = atoi(argv[c + 1]);
				sts.threadCount = std::max(-1, std::min<int>(std::thread::hardware_concurrency(), sts.threadCount));
				c++;
			}
		}
		return sts;
	}
};
