#pragma once

#if _WIN32
#define assert(test) do { ( (test) ? (void)0 : __debugbreak() )} while(0)
#else
#include <cassert>
#endif

#include <algorithm>
#include <mutex>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>

typedef std::lock_guard<std::mutex> lock_guard;
typedef std::unique_lock<std::mutex> unique_lock;

struct Settings {
	cv::CommandLineParser cmd;
	std::string videoPath;
	std::string termsFile;
	std::string resultDir;
	bool showFrame = true;
	bool silent = false;
	bool doCrop = false;
	bool verbose = false;
	int threadCount = -1;
	int matchLimit = 1;
	int frameSkip = 24;

	Settings(int argc, const char *const argv[], const std::string &format) : cmd(argc, argv, format) {}

	bool isValid() const;

	bool checkAndPrint() const;

	static Settings getSettings(int argc, char* argv[]);
};

using ms = std::chrono::milliseconds;

std::string timeToString(ms time);
