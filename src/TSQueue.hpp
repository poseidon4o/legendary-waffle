#pragma once

#include "Utils.hpp"

#include <vector>
#include <mutex>

template <typename Container>
struct ContainerLock {
	ContainerLock(Container &container, std::mutex &mtx)
		: mtx(mtx)
		, container(container) {
		mtx.lock();
	}

	~ContainerLock() {
		mtx.unlock();
	}

	Container &get() {
		return container;
	}

	ContainerLock(ContainerLock &&) = default;

	bool ownsLock(const std::mutex &other) const {
		return &mtx == &other;
	}

	ContainerLock(const ContainerLock &) = delete;
	ContainerLock &operator=(const ContainerLock &) = delete;
private:
	std::mutex &mtx;
	Container &container;
};

template <typename T>
struct Queue {
	typedef std::vector<T> ContainerType;
	typedef ContainerLock<ContainerType> Lock;

	Queue(int reserve = 0) {
		if (reserve) {
			data.reserve(reserve);
		}
	}

	Lock lock() {
		return Lock(data, mutex);
	}

	void push(const T &item, Lock &lock) {
		assertHolds(lock);
		data.push_back(item);
	}

	T pop(Lock &lock) {
		assertHolds(lock);
		T item = data.back();
		data.pop_back();
		return item;
	}

	int size(Lock &lock) const {
		assertHolds(lock);
		return int(data.size());
	}
private:
	void assertHolds(Lock &lock) const {
		if (!lock.ownsLock(mutex)) {
			assert("Using container with other lock" && false);
		}
	}
	ContainerType data;
	std::mutex mutex;
};

