#include "hittablelist.h"


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
	Material* closestMat = nullptr;
	
	for (int i = 0; i < objCount; i++) {
		Hittable* object = objects[i];
		cuda::std::optional<Intersection> tempInt = object->hit(ray, rayTMin, rayTMax);
		if (tempInt.has_value()) {
			const Intersection& result = tempInt.value();
			if (result.hitRec.t < closestDist) {
				hitAnything = true;
				closestObj = object;
				closestDist = result.hitRec.t;
				closestHit = result.hitRec;
				closestMat = result.mat;
			}
		}
	}
	if (!hitAnything) return HSRec;
	HSRec.hitRec = closestHit;
	if (closestMat != nullptr)
		HSRec.scatterRec = closestMat->scatter(ray, closestHit, rng);

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