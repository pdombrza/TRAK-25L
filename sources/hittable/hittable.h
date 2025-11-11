#pragma once

#include <glm/glm.hpp>
#include <cuda/std/optional>

#include "ray/ray.h"


template<typename T>
__host__ __device__ int sign(T val) {
	auto sign = (T(0) < val) - (T(0) > val);
	return sign;
}


struct HitRecord {
	glm::vec3 p{};
	glm::vec3 normal{};
	float t;
	bool frontFace;

	__device__ void setFaceNormal(const Ray& ray, const glm::vec3& outwardNormal) {
		// outwardNormal is supposed to be normalized
		frontFace = glm::dot(ray.getDirection(), outwardNormal) < 0;
		normal = frontFace ? outwardNormal : -outwardNormal;
	}
};


class Hittable {
protected:
	glm::vec3 center{};
public:
	__device__ virtual ~Hittable() = default;
	__device__ virtual cuda::std::optional<HitRecord> hit(const Ray& ray, float rayTMin, float rayTMax) const = 0;
	__device__ virtual HitRecord constructHitRecord(const Ray& ray, float t) const = 0;
	__device__ virtual glm::vec3 getCenter() const = 0;
};


class Sphere : public Hittable {
protected:
	glm::vec3 center{};
	float radius{};
public:
	__device__ ~Sphere() = default;
	__device__ explicit Sphere(const glm::vec3& center, float radius) : Hittable(), center(center), radius(fmaxf(0.0f, radius)) {};
	__device__ virtual cuda::std::optional<HitRecord> hit(const Ray& ray, float rayTMin, float rayTMax) const override;
	__device__ virtual HitRecord constructHitRecord(const Ray& ray, float t) const override;
	__device__ virtual glm::vec3 getCenter() const override;
};