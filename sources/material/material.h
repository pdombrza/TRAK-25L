#pragma once

#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <cuda/std/optional>
#include <cuda_runtime.h>

#include "hitrec/hitrec.h"
#include "ray/ray.h"
#include "utils/utils.h"
#include "scattering_record/scattering_record.h"


class Material {
public:
	__device__ virtual ~Material() = default;
	__device__ virtual cuda::std::optional<ScatteringRecord> scatter(const Ray& rayIn, const HitRecord& hitrec, utils::random::RNG& rng) const = 0;
};


class Lambertian : public Material {
private:
	glm::vec3 albedo{};
public:
	__device__ explicit Lambertian(const glm::vec3& albedo) : albedo(albedo) {};
	__device__ cuda::std::optional<ScatteringRecord> scatter(const Ray& ray_in, const HitRecord& hitrec, utils::random::RNG& rng) const override;
};


class Metal : public Material {
private:
	glm::vec3 albedo{};
	float fuzziness{1.0f};
public:
	__device__ explicit Metal(const glm::vec3& albedo, float fuzziness) : albedo(albedo), fuzziness(fminf(1.0f, fuzziness)) {};
	__device__ cuda::std::optional<ScatteringRecord> scatter(const Ray& rayIn, const HitRecord& hitrec, utils::random::RNG& rng) const override;
};


class Dielectric : public Material {
private:
	float refractionIndex{};
	__device__ static float reflectance(const float cosine, const float refractionIndex); // somehow produces wrong result when non-static
public:
	__device__ explicit Dielectric(float refractionIndex) : refractionIndex(refractionIndex) {};
	__device__ cuda::std::optional<ScatteringRecord> scatter(const Ray& rayIn, const HitRecord& hitrec, utils::random::RNG& rng) const override;
};