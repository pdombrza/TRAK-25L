#include "hittablelist.h"

#include <glm/glm.hpp>
#include <cuda/std/optional>

__device__ void HittableList::clear() {
	objCount = 0;
}

__device__ void HittableList::add(Hittable* hittable) {
	if (objCount < capacity) {
		objects[objCount] = hittable;
		objCount++;
	}
}

__device__ cuda::std::optional<HitRecord> HittableList::hit(const Ray& ray, float rayTMin, float rayTMax) const {
	HitRecord closestHit{};
	float closestDist = rayTMax;
	bool hitAnything = false;

	for (int i = 0; i < objCount; i++) {
		Hittable* object = objects[i];
		cuda::std::optional<HitRecord> tempRec = object->hit(ray, rayTMin, rayTMax);
		if (tempRec.has_value() && tempRec.value().t < closestDist) {
			hitAnything = true;
			closestDist = tempRec.value().t;
			closestHit = tempRec.value();
		}
	}
	if (hitAnything) {
		return cuda::std::optional<HitRecord>(closestHit);
	}
	return {};
}