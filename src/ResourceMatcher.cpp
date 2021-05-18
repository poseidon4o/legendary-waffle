#include "ResourceMatcher.h"

#include <fstream>
#include <algorithm>

ResourceMatcher::ResourceMatcher(const std::vector<std::string>& keywords): keywords(keywords) {}

void ResourceMatcher::addBlock(const CharPtr& data, const cv::Rect &where) {
	const int len = int(strlen(data.get()));

	for (const std::string &keyWord : keywords) {
		auto it = std::search(
			data.get(), data.get() + len,
			keyWord.begin(), keyWord.end(),
			[](char a, char b) {
				return std::tolower(a) == std::tolower(b);
			}
		);
		if (it != data.get() + len) {
			matches.push_back({keyWord, where});
			++found;
		}
	}
}

float ResourceMatcher::getMatchConfidence() const {
	return std::min(1.f, found / float(keywords.size()));
}

void ResourceMatcher::clear() {
	found = 0;
	matches.clear();
}

bool MatcherFactory::init() {
	std::fstream file(matchersFile, std::ios::in);
	if (!file) {
		return false;
	}
	std::vector<std::string> keyWords;
	std::string line;
	while(std::getline(file, line)) {
		std::stringstream stream(line);
		std::string word;
		while (stream >> word) {
			keyWords.push_back(word);
		}
		wordLists.push_back(keyWords);
		keyWords.clear();
	}
	return !wordLists.empty();
}

void MatcherFactory::showInfo() const {
	for (int c = 0; c < wordLists.size(); c++) {
		printf("Matcher[%d]:", c);
		for (const std::string& w : wordLists[c]) {
			printf("%s ", w.c_str());
		}
		puts("");
	}
}

void MatcherFactory::create(std::vector<ResourceMatcher>& matchers) const {
	matchers.clear();
	for (const std::vector<std::string>& keyWords : wordLists) {
		matchers.push_back(ResourceMatcher(keyWords));
	}
}
