#pragma once

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

typedef std::unique_ptr<char[]> CharPtr;

struct ResourceMatcher {
	ResourceMatcher(const std::vector<std::string> *keywords = nullptr, int required = -1);

	void addBlock(const CharPtr &data, const cv::Rect &where);

	float getMatchConfidence() const;

	bool isMatchFound() const;

	void clear();

	struct Match {
		std::string keyWord;
		std::string actual;
		int distance;
		cv::Rect bbox;
	};

	int required = -1;
	int found = 0;
	constexpr static float minThreshold = 0.3f;
	std::vector<Match> matches;

	const std::vector<std::string> *keywords = nullptr;
	std::vector<bool> used;
};

struct MatcherFactory {
	std::string matchersFile;
	struct Descriptor {
		int required = -1;
		std::vector<std::string> words;
	};
	std::vector<Descriptor> descriptors;

	bool init();

	void showInfo() const;

	void create(std::vector<ResourceMatcher>& matchers) const;
};
