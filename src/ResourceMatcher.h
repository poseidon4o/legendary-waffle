#pragma once

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

typedef std::unique_ptr<char> CharPtr;

struct ResourceMatcher {
	virtual void addBlock(const CharPtr &data, const cv::Rect &where) = 0;

	virtual float getMatchConfidence() {
		return 0.f; 
	}

	struct Match {
		std::string value;
		cv::Rect bbox;
	};

	std::vector<Match> matches;
};

struct DuckDuckGoMatcher : ResourceMatcher {
	void addBlock(const CharPtr& data, const cv::Rect &where) override;
	float getMatchConfidence() override;

	int found = 0;

	static const std::vector<std::string> keywords;
};

