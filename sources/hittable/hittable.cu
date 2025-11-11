#include "hittable.h"

__device__ HitRecord Sphere::constructHitRecord(const Ray& ray, float t) const {
	HitRecord rec{};
	rec.t = t;
	rec.p = ray.At(t);
	glm::vec3 outwardNormal = glm::normalize(rec.p - center);
	outwardNormal *= sign(radius);
	rec.setFaceNormal(ray, outwardNormal);
	return rec;
}

__device__ cuda::std::optional<HitRecord> Sphere::hit(const Ray& ray, float rayTMin, float rayTMax) const {
	glm::vec3 distOc = center - ray.getOrigin();
	float a = glm::dot(ray.getDirection(), ray.getDirection());
	float halfb = glm::dot(ray.getDirection(), distOc);
	float c = glm::dot(distOc, distOc) - radius * radius;
	auto discriminant = halfb * halfb - a * c;
	if (discriminant < 0) return {};

	float sqrtDiscriminant = sqrtf(discriminant);
	float root = (halfb - sqrtDiscriminant) / a;
	if (root <= rayTMin || root >= rayTMax) {
		root = (halfb + sqrtDiscriminant) / a;
		if (root < rayTMin || root >= rayTMax) {
			return {};
		}
	}
	HitRecord record = constructHitRecord(ray, root);
	return record;
}

__device__ glm::vec3 Sphere::getCenter() const {
	return center;
}