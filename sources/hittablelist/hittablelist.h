#pragma once

#include <cuda/std/optional>

#include "hittable/hittable.h"
#include "material/material.h"
#include "hitrec/hitrec.h"
#include "ray/ray.h"


class HittableList {
private:
	Hittable** objects = nullptr;
	int objCount = 0;
	int capacity = 0;
public:
	__device__ HittableList() = default;
	__device__ HittableList(Hittable** objectArray, int numObjects, int capacity) : objects(objectArray), objCount(numObjects), capacity(capacity) {};
	__device__ ~HittableList() = default;
	__device__ void clear();
	__device__ void add(Hittable* hittable);
	__device__ HitScatterRecord hit(const Ray& ray, float rayTMin, float rayTMax, utils::random::RNG& rng) const;
	__device__ int getObjCount() const { return objCount; };
	__device__ int getCapacity() const { return capacity; };
	__host__ __device__ Hittable** getObjects() const { return objects; };
	__host__ __device__ void setObjects(Hittable** newObjects) { objects = newObjects; };

	__device__ cuda::std::optional<AABB> boundingBox() const;
};