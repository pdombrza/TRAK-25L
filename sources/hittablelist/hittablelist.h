#pragma once 
#include <cuda_runtime.h>
#include <cuda/std/optional>

#include "ray/ray.h"
#include "hittable/hittable.h"

class HittableList {
private:
	Hittable** objects = nullptr;
	int capacity = 0;
	int objCount = 0;
public:
	__device__ HittableList() = default;
	__device__ HittableList(Hittable** objectArray, int capacity) : objects(objectArray), capacity(capacity), objCount(capacity) {};
	__device__ void clear();
	__device__ void add(Hittable* hittable);
	__device__ cuda::std::optional<HitRecord> hit(const Ray& ray, float rayTMin, float rayTMax) const;
};