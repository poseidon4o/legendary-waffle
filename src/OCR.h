#pragma once

#include "TSQueue.hpp"
#include "Utils.hpp"
#include "ResourceMatcher.h"

#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>

#include <thread>

/// Wrapper over cv::VideoCapture to allow easy access
struct VideoFile {
	bool init(const Settings& settings);

	cv::Mat getFrame(int index);

	~VideoFile();

	cv::VideoCapture video;
	int frameCount = 0;
};

/// Wrapper over TessBaseAPI
struct TesseractCTX {

	bool init(int idx);

	void orcImage(const cv::Mat& frame);

	int index = 0;
	tesseract::TessBaseAPI tesseract;
};

struct OCR {
	OCR(int frameIndex = -1, int totalFrames = -1);

	void processFrame(TesseractCTX& ctx, const cv::Mat& matchFrame);

	bool matchFound() const;

	bool showFrame = true;
	int frameIndex = -1;
	int totalFrames = -1;
	std::unique_ptr<ResourceMatcher> foundMatch;
	cv::Mat frame;
	std::vector<std::unique_ptr<ResourceMatcher>> matchers;
};

struct ThreadedOCR {
	ThreadedOCR(VideoFile& video);
	bool start(int count = -1);

	void stopThreads();

	struct ThreadStartContext {
		std::condition_variable cvar;
		std::mutex mtx;
		bool started = false;
		bool err = false;
	};

	void threadStart(ThreadStartContext& threadCtx, int idx);

	void waitFinish();

	OCR result;
	std::condition_variable resultCvar;
	std::mutex resultMutex;

	VideoFile &video;
	std::mutex videoMutex;
	std::atomic<bool> stopFlag = false;
	std::atomic<int> nextFrame;
	int maxFrame;
	const int frameSkip = 24;

	std::vector<std::thread> threads;
};