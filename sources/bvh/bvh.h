#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <cuda/std/optional>

#include "hittable/hittable.h"
#include "hittablelist/hittablelist.h"
#include "aabb/aabb.h"

class BVHNode : public Hittable {
private:
	Hittable* left;
	Hittable* right;
	AABB box;
public:
	__device__ BVHNode() = default;
	__device__ BVHNode(Hittable** objects, int start, int end, utils::random::RNG& rng);
	__device__ ~BVHNode() {
		delete left;
		if (right != left) {
			delete right;
		}
	};
	__device__ virtual cuda::std::optional<HitRecord> hit(const Ray& ray, float rayTMin, float rayTMax) const override;
	__device__ virtual AABB boundingBox() const override;
	__device__ virtual HitRecord constructHitRecord(const Ray& ray, float t) const { return {}; };
	__device__ virtual Material* getMaterial() const { return nullptr; };
};

__device__ inline bool boxXCompare(Hittable* a, Hittable* b) {
	AABB boxA = a->boundingBox();
	AABB boxB = b->boundingBox();
	return boxA.minPoint.x < boxB.minPoint.x;
};

__device__ inline bool boxYCompare(Hittable* a, Hittable* b) {
	AABB boxA = a->boundingBox();
	AABB boxB = b->boundingBox();
	return boxA.minPoint.y < boxB.minPoint.y;
};

__device__ inline bool boxZCompare(Hittable* a, Hittable* b) {
	AABB boxA = a->boundingBox();
	AABB boxB = b->boundingBox();
	return boxA.minPoint.z < boxB.minPoint.z;
};