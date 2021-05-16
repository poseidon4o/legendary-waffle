#include "ResourceMatcher.h"

const std::vector<std::string> SpecificMatcher::keywords = {
	"test1", "test2", "test3",
};

void SpecificMatcher::addBlock(const CharPtr& data, const cv::Rect &where) {
	for (const std::string &keyWord : keywords) {
		if (strstr(data.get(), keyWord.c_str())) {
			matches.push_back({keyWord, where});
			++found;
		}
	}
}

float SpecificMatcher::getMatchConfidence() {
	return std::min(1.f, found / float(keywords.size()));
}
