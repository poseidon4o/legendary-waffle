#include "OCR.h"

#include <leptonica/allheaders.h>
#include <tesseract/renderer.h>

#include <opencv2/imgproc/imgproc.hpp>

bool VideoFile::init(const Settings &settings) {
	video.open(settings.videoPath);
	if (!video.isOpened()) {
		return false;
	}

	frameCount = int(video.get(cv::CAP_PROP_FRAME_COUNT));
	return true;
}

void dbg(const cv::Mat &mat) {
	if (mat.cols > 1280 || mat.rows > 720) {
		cv::Mat dbgCopy;
		const float factor = float(1280) / mat.cols;
		cv::resize(mat, dbgCopy, cv::Size(), factor, factor);
		cv::imshow("dbg", dbgCopy);
	} else {
		cv::imshow("dbg", mat);
	}
	cv::waitKey(0);
}

cv::Mat VideoFile::getFrame(int index) {
	video.set(cv::CAP_PROP_POS_FRAMES, index);
	cv::Mat frame;
	video >> frame;
#if 0
	cv::Mat average = frame.clone();
	for (int c = 0; c < 100; c++) {
		cv::Mat next;
		video >> next;
		cv::Mat diff = (frame - next);
		diff = diff.mul(diff);
		cv::Scalar sum = cv::sum(diff) / double(frame.cols * next.rows);
		const double threshold = 1.;
		const bool isSame = sum[0] < threshold && sum[1] < threshold && sum[2] < threshold;
		if (!isSame) {
			break;
		}

		average = average * 0.7 + next * 0.3;
	}
	// video encoding does not introduce temporal noise - average over frames is useless
	cv::imshow("tm", cv::abs(average - frame) * 100); cv::waitKey(0);
	//cv::imshow("tm", average); cv::waitKey(0);
	return average;
#endif
	return frame;
}

int VideoFile::frameToMs(int index) {
	video.set(cv::CAP_PROP_POS_FRAMES, index);
	return int(video.get(cv::CAP_PROP_POS_MSEC));
}

VideoFile::~VideoFile() {
	video.release();
}

bool TesseractCTX::init(int idx) {
	index = idx;
	if (tesseract.Init(TESSDATA_DIR, "eng", tesseract::OEM_TESSERACT_LSTM_COMBINED) != 0) {
		return false;
	}
	tesseract.SetPageSegMode(tesseract::PSM_AUTO_OSD);
	char fileName[128] = {0,};
	snprintf(fileName, sizeof(fileName), "tesseract-%d.log", index);
	tesseract.SetVariable("tessedit_write_images", "1");
	tesseract.SetVariable("debug_file", fileName);
	return true;
}

void TesseractCTX::orcImage(const cv::Mat& frame) {
	tesseract.SetImage(frame.data, frame.cols, frame.rows, 1, int(frame.step));
	tesseract.Recognize(nullptr);
}

OCR::OCR(const MatcherFactory &factory, int totalFrames)
	: totalFrames(totalFrames)
{
	factory.create(matchers);
}

static void makePrintable(CharPtr &ptr) {
	int len = int(strlen(ptr.get()));
	int step = 0;
	for (int c = 0; c <= len; c++) {
		if (ptr[c] == '\t' || ptr[c] == '\n' || ptr[c] == '\r') {
			++step;
		} else {
			ptr[c - step] = ptr[c];
		}
	}
	len -= step;

	step = 0;
	for (int c = 0; c <= len; c++) {
		if (ptr[c] == ' ' && c < len && ptr[c + 1] == ' ') {
			step++;
		} else {
			ptr[c - step] = ptr[c];
		}
	}
}

static int clamp(int value, int min, int max) {
	return std::max(min, std::min(value, max));
}

cv::Rect operator/(const cv::Rect &rect, float factor) {
	cv::Rect res = rect;
	res.x = int(res.x / factor);
	res.y = int(res.y / factor);
	res.width = int(res.width / factor);
	res.height = int(res.height / factor);
	return res;
}

void OCR::processFrame(FrameProcessContext &ctx, const cv::Mat& sourceFrame) {
	frameIndex = ctx.frameIndex;
	const int percent = int(float(frameIndex) / totalFrames * 100);
	if (ctx.settings.verbose) {
		printf("Thread[%d]: Processing frame [%d/%d] %d%%\n", ctx.recognizer.index, frameIndex, totalFrames, percent);
		fflush(stdout);
	}
	const cv::Mat processed = preprocessFrame(ctx.settings, sourceFrame);
	const float factor = float(processed.cols) / sourceFrame.cols;
	ctx.recognizer.orcImage(processed);
#if 0
	Pix *thImage = recognizer.tesseract.GetThresholdedImage();
	char buff[64];
	snprintf(buff, sizeof(buff), "thimg-%d.png", frameNum);
	pixWriteAutoFormat(buff, thImage);
	pixDestroy(&thImage);
#endif

	tesseract::ResultIterator* iter = ctx.recognizer.tesseract.GetIterator();
	if (!iter) {
		assert(false);
		return;
	}

	const cv::Scalar red = {0, 0, 255};
	iter->Begin();
	do {
		if (iter->Empty(tesseract::RIL_PARA)) {
			continue;
		}
		CharPtr text(iter->GetUTF8Text(tesseract::RIL_PARA));
		const int len = int(strlen(text.get()));
		std::transform(text.get(), text.get() + len, text.get(), [] (char c) {
			return char(tolower(c));
		});

		int left, top, right, bottom;
		if (iter->BoundingBox(tesseract::RIL_PARA, &left, &top, &right, &bottom)) {
			const cv::Rect bbox = cv::Rect{{left, top}, cv::Size{right - left, bottom - top}} / factor;
			cv::rectangle(sourceFrame, bbox, {255, 0, 0});
			for (ResourceMatcher &matcher : matchers) {
				matcher.addBlock(text, bbox);
			}
		}
	} while (iter->Next(tesseract::RIL_PARA));

	std::string matchName;
	for (int c = 0; c < int(matchers.size()); c++) {
		if (!matchers[c].isMatchFound()) {
			continue;
		}
		if (matchName.empty()) {
			matchName = matchers[c].displayName;
		}
		for (const auto& match : matchers[c].matches) {
			cv::rectangle(sourceFrame, match.bbox, red);
		}
		matchIndices.push_back(c);
	}

	// dbg(sourceFrame);

	if (matchFound()) {
		// only first match frame is saved
		if (ctx.isFirstMatch.exchange(false) == true) {
			frame = sourceFrame;
		}

		if (!ctx.settings.resultDir.empty()) {
			char path[256]{0,};
			snprintf(path, sizeof(path), "%s/frame-%d.jpeg", ctx.settings.resultDir.c_str(), frameIndex);
			cv::imwrite(path, sourceFrame);
		}
		if (ctx.settings.verbose && ctx.settings.matchLimit > 1) {
			const int matchIndex = ctx.matchIndex.fetch_add(1);
			printf("Match found frame: [%d], [%s]  %d/%d\n", frameIndex, matchName.c_str(), matchIndex, ctx.settings.matchLimit);
			fflush(stdout);
		}
	}
}

bool OCR::matchFound() const {
	return !matchIndices.empty();
}

void OCR::clear() {
	matchIndices.clear();
	for (ResourceMatcher& matcher : matchers) {
		matcher.clear();
	}
	frame.release();
	frameIndex = -1;
}

cv::Mat OCR::preprocessFrame(const Settings &settings, cv::Mat input) const {
	cv::Mat processed = input.clone();
	if (settings.doCrop) {
		processed = processed.colRange(0, int(processed.cols / 2));
	}
	// zoom 2x to enable small text recognition
	cv::resize(input, processed, {}, 4., 4., cv::INTER_CUBIC);
	cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
	//dbg(processed);

	//cv::threshold(frame, frame, 128., 255., cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
	// cv::adaptiveThreshold(frame, frame, 255., cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 21, 2.);
	// cv::imshow("tm", frame); cv::waitKey(0);
	//cv::adaptiveThreshold(frame, frame, 255., cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, 2.);
	return processed;
}

ThreadedOCR::ThreadedOCR(const Settings &settings, const MatcherFactory &factory, VideoFile &video)
	: settings(settings)
	, factory(factory)
	, video(video)
	, maxFrame(video.frameCount)
	, remainingMatches(settings.matchLimit)
	, frameSkip(settings.frameSkip)
{}

bool ThreadedOCR::start(int count) {
	shouldStop = false;
	maxFrame = video.frameCount;
	const int threadCount = count == -1 ? std::thread::hardware_concurrency() : count;
	if (threadCount == 1) {
		ThreadStartContext ctx;
		threadStart(ctx, 0);
		return true;
	}
	for (int c = 0; c < threadCount; c++) {
		ThreadStartContext ctx;
		threads.push_back(std::thread(&ThreadedOCR::threadStart, this, std::ref(ctx), c));
		unique_lock lock(ctx.mtx);
		ctx.cvar.wait(lock, [&ctx]() {
			return ctx.started;
		});
		if (ctx.err) {
			shouldStop.store(true);
			break;
		}
	}

	if (shouldStop.load()) {
		stopThreads();
		return false;
	}
	return true;
}

void ThreadedOCR::stopThreads() {
	shouldStop.store(true);
	for (int c = 0; c < threads.size(); c++) {
		threads[c].join();
	}
}

void ThreadedOCR::threadStart(ThreadStartContext& threadCtx, int idx) {
	TesseractCTX tessCtx;
	runningThreads.fetch_add(1);
	const bool isInit = tessCtx.init(idx);
	{
		lock_guard lock(threadCtx.mtx);
		threadCtx.started = true;
		threadCtx.err = !isInit;
	}
	threadCtx.cvar.notify_one();

	if (!isInit) {
		return;
	}

	OCR ocr(factory, video.frameCount);

	int frameIdx = nextFrame.fetch_add(frameSkip);
	while (frameIdx < maxFrame) {
		if (shouldStop.load()) {
			return;
		}
		cv::Mat frame;
		{
			lock_guard lock(videoMutex);
			frame = video.getFrame(frameIdx);
		}
		ocr.clear();
		FrameProcessContext ctx {isFirstMatch, settings, tessCtx, frameIdx, matchIndex};
		ocr.processFrame(ctx, frame);

		if (ocr.matchFound()) {
			const int remaining = remainingMatches.fetch_sub(1);
			if (remaining >= 1) {
				lock_guard resLock(resultMutex);
				results.push_back(std::move(ocr));
			}
			if (remaining == 1) {
				shouldStop.store(true);
			}
		}

		if (shouldStop.load()) {
			resultCvar.notify_all();
			break;
		}

		frameIdx = nextFrame.fetch_add(frameSkip);
	}
	const int remaining = runningThreads.fetch_sub(1);
	if (remaining == 1) {
		resultCvar.notify_all();
	}
}

void ThreadedOCR::waitFinish() {
	{
		unique_lock lock(resultMutex);
		resultCvar.wait(lock, [this]() {
			return remainingMatches == int(results.size()) // all results found
				|| shouldStop.load() == true // stop flag has been set
				|| runningThreads.load() == 0; // all threads are done
		});
	}
	stopThreads();
}

bool ThreadedOCR::foundAnyMatches() const {
	return remainingMatches.load() < settings.matchLimit;
}
