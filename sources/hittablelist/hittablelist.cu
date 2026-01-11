#include <glm/glm.hpp>

#include "hittablelist.h"
#include "aabb/aabb.h"


__device__ void HittableList::clear() {
	objCount = 0;
}

__device__ void HittableList::add(Hittable* hittable) {
	if (objCount < capacity) {
		objects[objCount] = hittable;
		objCount++;
	}
}

__device__ HitScatterRecord HittableList::hit(const Ray& ray, float rayTMin, float rayTMax, utils::random::RNG& rng) const {
	HitRecord closestHit;
	HitScatterRecord HSRec{};
	float closestDist = rayTMax;
	bool hitAnything = false;
	Hittable* closestObj = nullptr;
	
	for (int i = 0; i < objCount; i++) {
		Hittable* object = objects[i];
		cuda::std::optional<HitRecord> tempRec = object->hit(ray, rayTMin, rayTMax);
		if (tempRec.has_value() && tempRec.value().t < closestDist) {
			hitAnything = true;
			closestObj = object;
			closestDist = tempRec.value().t;
			closestHit = tempRec.value();
		}
	}
	if (!hitAnything) return HSRec;
	HSRec.hitRec = closestHit;
	cuda::std::optional<ScatteringRecord> sRec = closestObj->getMaterial()->scatter(ray, closestHit, rng);
	HSRec.scatterRec = sRec;

	return HSRec;
}

__device__ cuda::std::optional<AABB> HittableList::boundingBox() const {
	if (objCount == 0) {
		return {};
	}
	
	AABB tempBox;
	bool firstBox = true;
	
	for (int i = 0; i < objCount; i++) {
		Hittable* object = objects[i];
		AABB objBox = object->boundingBox();
		if (firstBox) {
			tempBox = objBox;
			firstBox = false;
		}
		else {
			tempBox = joinAABBs(tempBox, objBox);
		}
	}
	return tempBox;
}