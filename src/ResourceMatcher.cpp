#include "ResourceMatcher.h"

#include <fstream>
#include <algorithm>

static int getEditDistance(const char * word1, int len1, const char * word2, int len2) {
	typedef std::vector<int> IntArr;
	typedef std::vector<IntArr> Table;
	Table matrix(len1 + 1, IntArr(len2 + 1));
	int i;
	for (i = 0; i <= len1; i++) {
		matrix[i][0] = i;
	}
	for (i = 0; i <= len2; i++) {
		matrix[0][i] = i;
	}
	for (i = 1; i <= len1; i++) {
		int j;
		char c1;

		c1 = word1[i - 1];
		for (j = 1; j <= len2; j++) {
			char c2;

			c2 = word2[j - 1];
			if (c1 == c2) {
				matrix[i][j] = matrix[i - 1][j - 1];
			} else {
				int del = matrix[i - 1][j] + 1;
				int insert = matrix[i][j - 1] + 1;
				int substitute = matrix[i - 1][j - 1] + 1;
				int minimum = del;
				if (insert < minimum) {
					minimum = insert;
				}
				if (substitute < minimum) {
					minimum = substitute;
				}
				matrix[i][j] = minimum;
			}
		}
	}
	return matrix[len1][len2];
}


ResourceMatcher::ResourceMatcher(const Descriptor &descriptor)
	: required(descriptor.required)
	, displayName(descriptor.name)
	, keywords(&descriptor.words)
	, used(keywords->size(), false)
{}

void ResourceMatcher::addBlock(const CharPtr& data, const cv::Rect &where) {
	const int len = int(strlen(data.get()));
	if (!len) {
		return;
	}

	for (int c = 0; c < keywords->size(); c++) {
		const std::string &keyWord = (*keywords)[c];
		if (used[c]) {
			continue;
		}
#ifdef WITH_EDIT_DISTANCE
		int bestDistance = INT_MAX;
		Match match;
		for (int c = 0; c < len; c++) {
			const int inputLength = std::min(len - c, int(keyWord.size()));
			const int distance = getEditDistance(data.get() + c, inputLength, keyWord.c_str(), int(keyWord.size()));
			if (distance < bestDistance) {
				bestDistance = distance;
				match = {keyWord, std::string(data.get() + c, inputLength), distance, where};
			}
		}

		if (bestDistance <= 1) {
			++found;
			matches.push_back(match);
		}
#else
		char *end = data.get() + len;
		int matchLen = int(keyWord.length());
		char *it = std::search(data.get(), end, keyWord.begin(), keyWord.end());
		int distance = 0;
		if (keyWord.length() >= 5) {
			distance = 1;
			--matchLen;
			// try without first or last letter
			if (it == end) {
				it = std::search(data.get(), end, keyWord.begin() + 1, keyWord.end());
			}
			if (it == end) {
				it = std::search(data.get(), end, keyWord.begin(), keyWord.end() - 1);
			}
		}
		if (it != end) {
			matches.push_back({keyWord, std::string(it, matchLen), distance, where});
			++found;
			used[c] = true;
		}
#endif
	}
}

float ResourceMatcher::getMatchConfidence() const {
	const int kwSize = keywords ? int(keywords->size()) : INT_MAX;
	if (required == -1) {
		return std::min(1.f, found / float(kwSize));
	} else {
		return found >= required ? 1.0f : 0.0f;
	}
}

bool ResourceMatcher::isMatchFound() const {
	return getMatchConfidence() >= minThreshold;
}

void ResourceMatcher::clear() {
	found = 0;
	matches.clear();
	std::fill(used.begin(), used.end(), false);
}

bool MatcherFactory::init() {
	std::fstream file(matchersFile, std::ios::in);
	if (!file) {
		return false;
	}
	std::vector<std::string> keyWords;
	std::string line;
	while(std::getline(file, line)) {
		Descriptor desc;
		std::stringstream stream(line);
		std::string word;
		int required = -1;
		if (stream >> required) {
			desc.required = required;
		} else {
			stream.clear();
			stream.seekg(0);
		}
		while (stream >> word) {
			if (word.front() == '#') {
				if (word.size() == 1) {
					stream >> word;
				}
				if (stream && word != "#") {
					if (word.front() == '#') {
						word.erase(word.begin(), word.begin() + 1);
					}
					desc.name = word;
				}
				break;
			}
			keyWords.push_back(word);
		}
		desc.words.swap(keyWords);
		if (!desc.words.empty()) {
			if (desc.words.size() == 1) {
				desc.required = 1;
			}
			descriptors.push_back(desc);
		}
	}
	return !descriptors.empty();
}

void MatcherFactory::showInfo() const {
	for (int c = 0; c < descriptors.size(); c++) {
		if (descriptors[c].name.empty()) {
			printf("Matcher[%d]: ", c);
		} else {
			printf("Matcher[%s]: ", descriptors[c].name.c_str());
		}

		for (const std::string& w : descriptors[c].words) {
			printf("%s ", w.c_str());
		}
		puts("");
	}
}

void MatcherFactory::create(std::vector<ResourceMatcher>& matchers) const {
	matchers.clear();
	matchers.reserve(descriptors.size());
	for (const Descriptor &desc: descriptors) {
		matchers.emplace_back(desc);
	}
}
