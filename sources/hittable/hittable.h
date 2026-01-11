#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include <cuda/std/optional>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "ray/ray.h"
#include "hitrec/hitrec.h"
#include "intersection/intersection.h"
#include "material/material.h"
#include "aabb/aabb.h"


template<typename T>
__host__ __device__ int sign(T val) {
	auto sign = (T(0) < val) - (T(0) > val);
	return sign;
}


class Hittable {
protected:
	glm::vec3 center{};
public:
	__device__ virtual ~Hittable() = default;
	__device__ virtual cuda::std::optional<Intersection> hit(const Ray& ray, float rayTMin, float rayTMax) const = 0;
	__device__ virtual Intersection constructIntersection(const Ray& ray, float t) const = 0;

	__device__ virtual AABB boundingBox() const = 0;
};


class Sphere : public Hittable {
protected:
	Material* material = nullptr;
	glm::vec3 center{};
	float radius{};
public:
	__device__ ~Sphere() { if(material != nullptr) delete material; };
	__device__ explicit Sphere(const glm::vec3& center, float radius, Material* mat) : Hittable(), center(center), radius(fmaxf(0.0f, radius)), material(mat) {};

	__device__ virtual cuda::std::optional<Intersection> hit(const Ray& ray, float rayTMin, float rayTMax) const override {
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
		Intersection intersection = constructIntersection(ray, root);
		return intersection;
	};

	__device__ virtual Intersection constructIntersection(const Ray& ray, float t) const override {
		Intersection intersection{};
		HitRecord rec{};
		rec.t = t;
		rec.p = ray.At(t);
		glm::vec3 outwardNormal = glm::normalize(rec.p - center);
		outwardNormal *= sign(radius);
		rec.setFaceNormal(ray, outwardNormal);
		intersection.hitRec = rec;
		intersection.mat = material;

		return intersection;
	};

	__device__ virtual void setMaterial(Material* mat) {
		material = mat;
	};

	__device__ virtual Material* getMaterial() const {
		return material;
	};

	__device__ virtual glm::vec3 getCenter() const {
		return center;
	};

	__device__ virtual AABB boundingBox() const override {
		return AABB(
			center - glm::vec3(radius, radius, radius),
			center + glm::vec3(radius, radius, radius)
		);
	};
};