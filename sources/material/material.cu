#include "material.h"


__device__ cuda::std::optional<ScatteringRecord> Lambertian::scatter(const Ray& rayIn, const HitRecord& rec, utils::random::RNG& rng) const {
	glm::vec3 scatterDirection = rec.normal + rng.randomVec3InSphere();

	// Catch degenerate scatter direction
	if (glm::all(glm::epsilonEqual(scatterDirection, glm::vec3(0.0f, 0.0f, 0.0f), cuda::std::numeric_limits<float>::epsilon())))
		scatterDirection = rec.normal;

	Ray scattered = Ray(rec.p, scatterDirection);
	return ScatteringRecord{scattered, albedo};
}


__device__ cuda::std::optional<ScatteringRecord> Metal::scatter(const Ray& rayIn, const HitRecord& rec, utils::random::RNG& rng) const {
	glm::vec3 reflected = glm::reflect(rayIn.getDirection(), rec.normal) + fuzziness * rng.randomVec3InSphere();
	Ray scattered = Ray(rec.p, reflected);
	if (glm::dot(reflected, rec.normal) > 0) {
		ScatteringRecord sRec{};
		sRec.ray = scattered;
		sRec.attenuation = albedo;
		return sRec;
	}
	return {};
}


__device__ cuda::std::optional<ScatteringRecord> Dielectric::scatter(const Ray& rayIn, const HitRecord& rec, utils::random::RNG& rng) const {
	glm::vec3 attenuation{ 1.0f, 1.0f, 1.0f };
	const float refractionRatio = rec.frontFace ? (1.0f / refractionIndex) : refractionIndex;

	glm::vec3 dir = rayIn.getDirection(); // already unit vector
	const float cosTheta = fminf(glm::dot(-1.0f * dir, rec.normal), 1.0f);
	const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	bool cannotRefract = refractionRatio * sinTheta > 1.0f;
	glm::vec3 scatterDirection{};

	if (cannotRefract || reflectance(cosTheta, refractionRatio) > rng.getRandomUniform())
		scatterDirection = glm::reflect(dir, rec.normal);
	else
		scatterDirection = glm::refract(dir, rec.normal, refractionRatio);

	Ray scattered = Ray(rec.p, scatterDirection);
	return ScatteringRecord{ scattered, attenuation };
}


__device__ float Dielectric::reflectance(const float cosine, const float refractionIndex) {
	// Schlick's approximation for reflectance
	float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}