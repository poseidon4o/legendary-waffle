#include "ResourceMatcher.h"

const std::vector<std::string> DuckDuckGoMatcher::keywords = {
	"duckduckgo", "All", "Images", "Videos", "News", "Maps", "Settings", "Privacy", "simplified",
};

void DuckDuckGoMatcher::addBlock(const CharPtr& data, const cv::Rect &where) {
	for (const std::string &keyWord : keywords) {
		if (strstr(data.get(), keyWord.c_str())) {
			matches.push_back({keyWord, where});
			++found;
		}
	}
}

float DuckDuckGoMatcher::getMatchConfidence() {
	return std::min(1.f, found / float(keywords.size()));
}
