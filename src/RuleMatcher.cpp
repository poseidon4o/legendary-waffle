#include "RuleMatcher.h"

#include <fstream>
#include <algorithm>

static int getEditDistance(const char *word1, int len1, const char *word2, int len2) {
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


CharPtrView::CharPtrView(CharPtr &&ptr): ptr(std::move(ptr)) {
	assert(ptr);
	if (ptr) {
		length = int(strlen(ptr.get()));
	}
}

CharPtrView::CharPtrView(CharPtr&& ptr, int length): ptr(std::move(ptr)), length(length) {
	assert(this->ptr.get());
}

RuleMatcher::RuleMatcher(const Descriptor &descriptor)
	: descriptorPtr(&descriptor)
	, used(descriptor.words.size(), false)
{}

void RuleMatcher::addBlock(const CharPtrView &data, const cv::Rect &where) {
	assert(descriptorPtr);
	if (!descriptorPtr) {
		return;
	}
	
	for (int c = 0; c < descriptorPtr->words.size(); c++) {
		const std::string &keyWord = descriptorPtr->words[c];
		if (used[c]) {
			continue;
		}

		TermMatch match;
		if (tryMatchWord(data, keyWord, match)) {
			match.bbox = where;
			matches.push_back(match);
			++found;
			used[c] = true;
		}

	}
}

bool RuleMatcher::isFullMatch(const CharPtrView &data, const cv::Rect &) {
	assert(descriptorPtr);
	if (!descriptorPtr) {
		return false;
	}

	TermMatch m;
	int count = 0;
	std::fill(used.begin(), used.end(), false);
	for (int c = 0; c < int(descriptorPtr->words.size()); c++) {
		if (used[c]) {
			continue;;
		}
		if (tryMatchWord(data, descriptorPtr->words[c], m)) {
			used[c] = true;
			++count;
		}
	}
	return count == descriptorPtr->required;
}

float RuleMatcher::getMatchConfidence() const {
	assert(descriptorPtr);
	if (!descriptorPtr) {
		return 0;
	}
	if (descriptorPtr->required == -1) {
		return descriptorPtr->words.size() == found ? 1.f : 0.f;
		// return std::min(1.f, found / float(descriptorPtr->words.size()));
	} else {
		return found >= descriptorPtr->required ? 1.f : 0.f;
	}
}

bool RuleMatcher::isMatchFound() const {
	return getMatchConfidence() >= minThreshold;
}

void RuleMatcher::clear() {
	found = 0;
	matches.clear();
	std::fill(used.begin(), used.end(), false);
}

bool RuleMatcher::tryMatchWord(const CharPtrView &data, const std::string& keyWord, TermMatch& match) {
#ifdef WITH_EDIT_DISTANCE
	int bestDistance = INT_MAX;
	for (int c = 0; c < len; c++) {
		const int inputLength = std::min(len - c, int(keyWord.size()));
		const int distance = getEditDistance(data.get() + c, inputLength, keyWord.c_str(), int(keyWord.size()));
		if (distance < bestDistance) {
			bestDistance = distance;
			match = {keyWord, std::string(data.get() + c, inputLength), distance, where};
		}
	}

	if (bestDistance <= 1) {
		return true;
	}
#else
	const char *end = data.get() + data.size();
	int matchLen = int(keyWord.length());
	const char *it = std::search(data.get(), end, keyWord.begin(), keyWord.end());
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
		match = {keyWord, std::string(it, matchLen), distance};
		return true;
	}
	return false;
#endif
}


void RuleSet::addBlock(const CharPtrView& data, const cv::Rect& where) {
	for (RuleMatcher &matcher : blacklist) {
		if (matcher.isFullMatch(data, where)) {
			return;
		}
	}

	for (RuleMatcher &matcher : whitelist) {
		matcher.addBlock(data, where);
	}
}

void RuleSet::clear() {
	for (RuleMatcher& matcher : whitelist) {
		matcher.clear();
	}
}

bool RuleSet::isEmpty() const {
	return whitelist.empty();
}

bool MatcherFactory::init() {
	std::fstream file(matchersFile, std::ios::in);
	if (!file) {
		return false;
	}
	
	std::string line;
	int lineIdx = 0;
	int ruleIdx = 0;
	while(std::getline(file, line)) {
		Descriptor desc;
		std::stringstream stream(line);
		std::string word;

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
			if (word == "-" && desc.words.empty()) {
				desc.isBlackList = true;
				continue;
			}
			if (word == "~" && desc.words.empty()) {
				desc.isSoftMatch = true;
				continue;
			}
			int required = 0;
			if (sscanf(word.c_str(), "%d", &required) == 1) {
				desc.required = required;
				continue;
			}

			desc.words.push_back(word);
		}
		desc.words.swap(desc.words);
		if (!desc.words.empty()) {
			if (desc.name.empty()) {
				char nameBuff[32];
				snprintf(nameBuff, sizeof(nameBuff), "#%d", ruleIdx);
				desc.name = nameBuff;
			}
			if (desc.isSoftMatch && desc.isBlackList) {
				printf("Rule on line %d can't be both soft match and WhiteList, fallback to just whitelist\n", lineIdx);
				desc.isSoftMatch = false;
			}
			if (desc.words.size() == 1) {
				desc.required = 1;
			}
			desc.required = std::min(desc.required, int(desc.words.size()));
			descriptors.push_back(desc);
			++ruleIdx;
		}
		line.clear();
		++lineIdx;
	}
	return !descriptors.empty();
}

void MatcherFactory::showInfo() const {
	for (int c = 0; c < descriptors.size(); c++) {
		const char *ruleMod = "";
		if (descriptors[c].isSoftMatch) {
			ruleMod = "~";
		} else if (descriptors[c].isBlackList) {
			ruleMod = "-";
		}
		printf("Matcher[%s%s]: ", ruleMod, descriptors[c].name.c_str());

		for (const std::string& w : descriptors[c].words) {
			printf("%s ", w.c_str());
		}
		puts("");
	}
}

void MatcherFactory::create(RuleSet &ruleSet) const {
	ruleSet.whitelist.clear();
	ruleSet.blacklist.clear();
	ruleSet.whitelist.reserve(descriptors.size());
	for (const Descriptor &desc: descriptors) {
		if (desc.isBlackList) {
			ruleSet.blacklist.push_back(desc);
		} else {
			ruleSet.whitelist.push_back(desc);
		}
	}
}
