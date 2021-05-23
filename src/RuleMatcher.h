#pragma once

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

typedef std::unique_ptr<char[]> CharPtr;

struct CharPtrView {
	CharPtr ptr;
	int length = 0;

	CharPtrView() = default;
	CharPtrView(CharPtr&& ptr);

	CharPtrView(CharPtr&& ptr, int length);

	operator const char *() const {
		assert(ptr);
		return ptr.get();
	}

	const char *get() const {
		return ptr.get();
	}

	int size() const {
		assert(ptr);
		return length;
	}
};

struct Descriptor {
	std::string name;
	int required = -1;
	bool isSoftMatch = false;
	bool isBlackList = false;
	std::vector<std::string> words;
};

struct TermMatch {
	std::string keyWord;
	std::string actual;
	int distance;
	cv::Rect bbox;
};

typedef std::vector<TermMatch> RuleMatch;

struct RuleMatcher {

	RuleMatcher() = default;
	RuleMatcher(const Descriptor &descriptor);

	void addBlock(const CharPtrView &data, const cv::Rect &where);

	bool isFullMatch(const CharPtrView &data, const cv::Rect &where);
	
	float getMatchConfidence() const;

	bool isMatchFound() const;

	void clear();

	const Descriptor &descriptor() const {
		assert(descriptorPtr);
		return *descriptorPtr;
	}

	const RuleMatch &getMatchedTerms() const {
		return matches;
	}
private:	

	bool tryMatchWord(const CharPtrView &data, const std::string &keyWord, TermMatch &match);

	int found = 0;
	constexpr static float minThreshold = 0.3f;
	RuleMatch matches;

	const Descriptor *descriptorPtr = nullptr;

	std::vector<bool> used;
};

typedef std::vector<RuleMatcher> MatcherList;

struct RuleSet {
	friend struct MatcherFactory;
	void addBlock(const CharPtrView &data, const cv::Rect &where);

	const MatcherList &getWhitelist() const {
		return whitelist;
	}

	const MatcherList &getBlacklist() const {
		return blacklist;
	}

	void clear();

	bool isEmpty() const;
private:
	MatcherList blacklist; ///< Rules that disqualify a block from matching anything
	MatcherList whitelist; ///< Actual rules to match
};

struct MatcherFactory {
	std::string matchersFile;
	std::vector<Descriptor> descriptors;

	bool init();

	void showInfo() const;

	void create(RuleSet& ruleSet) const;
};
