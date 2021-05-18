#pragma once

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

typedef std::unique_ptr<char[]> CharPtr;

struct ResourceMatcher {
	ResourceMatcher(const std::vector<std::string>& keywords);

	void addBlock(const CharPtr &data, const cv::Rect &where);

	float getMatchConfidence() const;

	void clear();

	struct Match {
		std::string value;
		cv::Rect bbox;
	};

	int found = 0;
	constexpr static float minThreshold = 0.3f;
	std::vector<Match> matches;
	const std::vector<std::string> keywords;
};

struct MatcherFactory {
	std::string matchersFile;
	std::vector<std::vector<std::string>> wordLists;

	bool init();

	void showInfo() const;

	void create(std::vector<ResourceMatcher>& matchers) const;
};
