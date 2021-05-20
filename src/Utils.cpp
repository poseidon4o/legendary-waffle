#include "Utils.h"

static const std::string ARGS_TEMPLATE =
"{ help h usage    |        | Print this message }"
"{ v video         |        | Path to video file to analyze }"
"{ t terms         |        | Path to file containing search terms }"
"{ resultDir       |        | If path to directory, saves all matching frames up to matchLimit }"
"{ show            | 1      | Show frame where first detection is found }"
"{ silent          | 0      | Print only on error and match found }"
"{ crop            | 0      | Crop image to upper/left 1/4th }"
"{ verbose         | 0      | If set to true will write progress messages }"
"{ threadCount     | -1     | Number of threads }"
"{ matchLimit      | 1      | Number of matches before matching stops }"
"{ frameSkip       | 24     | Number of frames to skip }";


bool Settings::isValid() const {
	return !videoPath.empty() && !termsFile.empty();
}

bool Settings::checkAndPrint() const {
	if (cmd.has("help")) {
		cmd.printMessage();
		return false;
	}
	if (!cmd.check()) {
		cmd.printErrors();
		return false;
	}
	return true;
}


Settings Settings::getSettings(int argc, char* argv[]) {
	Settings sts(argc, argv, ARGS_TEMPLATE);

	try {
		sts.videoPath = sts.cmd.get<cv::String>("video");
		sts.termsFile = sts.cmd.get<cv::String>("terms");
		sts.resultDir = sts.cmd.get<cv::String>("resultDir");

		sts.showFrame = sts.cmd.get<bool>("show");
		sts.silent = sts.cmd.get<bool>("silent");
		sts.doCrop = sts.cmd.get<bool>("crop");
		sts.verbose = sts.cmd.get<bool>("verbose");

		sts.threadCount = sts.cmd.get<int>("threadCount");
		sts.matchLimit = sts.cmd.get<int>("matchLimit");
		sts.frameSkip = sts.cmd.get<int>("frameSkip");
	} catch (cv::Exception &ex) {
		puts(ex.what());
	}

	return sts;
}
