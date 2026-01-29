#include "bvh.h"

__device__ BVHNode::BVHNode(Hittable** objects, int start, int end) {
	auto bbox = AABB();
	for (int i = start; i < end; i++) {
		bbox = joinAABBs(bbox, objects[i]->boundingBox());
	}
	int axis = bbox.longestAxis();

	auto comparator = (axis == 0) ? boxXCompare
					: (axis == 1) ? boxYCompare
								  : boxZCompare;
	int objectSpan = end - start;
	if (objectSpan == 1) {
		left = right = objects[start];
	}
	else if (objectSpan == 2) {
		if (comparator(objects[start], objects[start + 1])) {
			left = objects[start];
			right = objects[start + 1];
		}
		else {
			left = objects[start + 1];
			right = objects[start];
		}
	}
	else {
		thrust::sort(thrust::seq, objects + start, objects + end, comparator);
		int mid = start + objectSpan / 2;
		left = new BVHNode(objects, start, mid);
		right = new BVHNode(objects, mid, end);
	}
	AABB boxLeft = left->boundingBox();
	AABB boxRight = right->boundingBox();
	box = joinAABBs(boxLeft, boxRight);
}

__device__ cuda::std::optional<Intersection> BVHNode::hit(const Ray& ray, float rayTMin, float rayTMax) const {
	if (!box.hit(ray, rayTMin, rayTMax)) {
		return {};
	}
	cuda::std::optional<Intersection> leftHit = left->hit(ray, rayTMin, rayTMax);
	cuda::std::optional<Intersection> rightHit = right->hit(ray, rayTMin, rayTMax);
	if (leftHit.has_value() && rightHit.has_value()) {
		return (leftHit->hitRec.t < rightHit->hitRec.t) ? leftHit : rightHit;
	}
	else if (leftHit.has_value()) {
		return leftHit;
	}
	else {
		return rightHit;
	}
}

__device__ AABB BVHNode::boundingBox() const {
	return box;
}