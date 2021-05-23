#pragma once

#include "Utils.h"
#include "RuleMatcher.h"

#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <thread>

/// Wrapper over cv::VideoCapture to allow easy access
struct VideoFile {
	bool init(const Settings& settings);

	cv::Mat getFrame(int index);
	cv::Mat getFrame(int index, ms &frameTime);

	ms frameToMs(int index);

	~VideoFile();

	cv::VideoCapture video;
	int frameCount = 0;
};

/// Wrapper over TessBaseAPI
struct TesseractCTX {

	bool init(int idx);

	void orcImage(const cv::Mat &frame);

	int index = 0;
	tesseract::TessBaseAPI tesseract;
};

struct FrameProcessContext {
	std::atomic<bool> &isFirstMatch;
	const Settings &settings;
	TesseractCTX &recognizer;
	int frameIndex;
	std::atomic<int> &matchIndex;
};

struct MatchResult {
	enum MatchType {
		NoMatch = 0,
		SoftMatch = 1 << 0,
		HardMatch = 1 << 1,
	};

	std::vector<int> whitelistIndices;
	cv::Mat frame;
	MatchType matchType = NoMatch;
	int frameIndex = -1;
	RuleSet ruleSet;
};

struct OCR {
	OCR(OCR &&) = default;
	OCR &operator=(OCR &&) = default;
	OCR() = default;

	OCR(const MatcherFactory &factory, int totalFrames = -1);

	void processFrame(FrameProcessContext &ctx, const cv::Mat &frameData, ms frameTime);
	void clear();

	cv::Mat preprocessFrame(const Settings &settings, cv::Mat input) const;

	int totalFrames = -1;
	RuleSet ruleSet;

	MatchResult result;
};

struct ThreadedOCR {
	ThreadedOCR(const Settings &settings, const MatcherFactory &factory, VideoFile &video);
	bool start(int count = -1);

	void stopThreads();

	struct ThreadStartContext {
		std::condition_variable cvar;
		std::mutex mtx;
		bool started = false;
		bool err = false;
	};

	void threadStart(ThreadStartContext &threadCtx, int idx);

	void waitFinish();

	bool foundAnyMatches() const;

	const Settings settings;
	const MatcherFactory &factory;
	VideoFile &video;
	std::mutex videoMutex;

	std::vector<MatchResult> results;
	std::atomic<int> remainingMatches = 1;
	std::condition_variable resultCvar;
	std::mutex resultMutex;

	std::atomic<bool> shouldStop = false;
	std::atomic<int> runningThreads = 0;
	std::atomic<int> nextFrame = 0;

	// for FrameProcessContext
	std::atomic<bool> isFirstMatch = true;
	std::atomic<int> matchIndex = 0;

	int maxFrame;
	const int frameSkip = 24;

	std::vector<std::thread> threads;
};