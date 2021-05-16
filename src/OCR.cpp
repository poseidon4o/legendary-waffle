#include "OCR.h"

bool VideoFile::init(const Settings &settings) {
	video.open(settings.videoPath);
	if (!video.isOpened()) {
		return false;
	}

	frameCount = video.get(cv::CAP_PROP_FRAME_COUNT);
	return true;
}

cv::Mat VideoFile::getFrame(int index) {
	video.set(cv::CAP_PROP_POS_FRAMES, index);
	cv::Mat frame;
	video >> frame;
	return frame;
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
	tesseract.SetVariable("debug_file", fileName);
	return true;
}

void TesseractCTX::orcImage(const cv::Mat& frame) {
	tesseract.SetImage(frame.data, frame.cols, frame.rows, 3, frame.step);
	tesseract.Recognize(nullptr);
}

OCR::OCR(const MatcherFactory &factory, int totalFrames)
	: totalFrames(totalFrames)
{
	factory.create(matchers);
}

void OCR::processFrame(TesseractCTX& ctx, const cv::Mat& matchFrame, int frameIndex) {
	this->frameIndex = frameIndex;
	const int percent = int(float(frameIndex) / totalFrames * 100);
	printf("Thread[%d]: Processing frame [%d/%d] %d%%\n", ctx.index, frameIndex, totalFrames, percent);
	fflush(stdout);
	ctx.orcImage(matchFrame);

	tesseract::ResultIterator* iter = ctx.tesseract.GetIterator();
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

		int left, top, right, bottom, padding;
		if (iter->BoundingBox(tesseract::RIL_PARA, &left, &top, &right, &bottom)) {
			const cv::Rect bbox{{left, top}, cv::Size{right - left, bottom - top}};
			// cv::rectangle(matchFrame, bbox, {255, 0, 0});
			for (ResourceMatcher &matcher : matchers) {
				matcher.addBlock(text, bbox);
			}
		}
	} while (iter->Next(tesseract::RIL_PARA));

	for (ResourceMatcher &matcher : matchers) {
		if (matcher.getMatchConfidence() > ResourceMatcher::minThreshold) {
			frame = matchFrame;
			for (const auto& match : matcher.matches) {
				cv::rectangle(matchFrame, match.bbox, red);
			}
			foundMatch = std::make_unique<ResourceMatcher>(matcher);
			matchers.clear();
			break;
		}
	}
}

bool OCR::matchFound() const {
	return foundMatch != nullptr;
}

void OCR::clear() {
	foundMatch.reset(nullptr);
	for (ResourceMatcher& matcher : matchers) {
		matcher.clear();
	}
	frame.release();
	frameIndex = -1;
}

ThreadedOCR::ThreadedOCR(const Settings &settings, const MatcherFactory &factory, VideoFile &video)
	: settings(settings)
	, factory(factory)
	, frameSkip(settings.frameSkip)
	, video(video)
	, result(factory)
{}

bool ThreadedOCR::start(int count) {
	stopFlag = false;
	maxFrame = video.frameCount;
	const int threadCount = count == -1 ? std::thread::hardware_concurrency() : count;
	for (int c = 0; c < threadCount; c++) {
		ThreadStartContext ctx;
		threads.push_back(std::thread(&ThreadedOCR::threadStart, this, std::ref(ctx), c));
		unique_lock lock(ctx.mtx);
		ctx.cvar.wait(lock, [&ctx]() {
			return ctx.started;
		});
		if (ctx.err) {
			stopFlag.store(true);
			break;
		}
	}

	if (stopFlag.load()) {
		stopThreads();
		return false;
	}
	return true;
}

void ThreadedOCR::stopThreads() {
	stopFlag.store(true);
	for (int c = 0; c < threads.size(); c++) {
		threads[c].join();
	}
}

void ThreadedOCR::threadStart(ThreadStartContext& threadCtx, int idx) {
	TesseractCTX tessCtx;
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
		if (stopFlag.load()) {
			return;
		}
		cv::Mat frame;
		{
			lock_guard lock(videoMutex);
			frame = video.getFrame(frameIdx);
		}
		ocr.clear();
		ocr.processFrame(tessCtx, frame, frameIdx);

		if (ocr.matchFound()) {
			lock_guard resLock(resultMutex);
			result = std::move(ocr);
			stopFlag = true;
		}

		if (stopFlag.load()) {
			resultCvar.notify_all();
			return;
		}

		frameIdx = nextFrame.fetch_add(frameSkip);
	}
	resultCvar.notify_all();
}

void ThreadedOCR::waitFinish() {
	{
		unique_lock lock(resultMutex);
		resultCvar.wait(lock, [this]() {
			return result.matchFound() || stopFlag;
		});
	}
	stopThreads();
}
