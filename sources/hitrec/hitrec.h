#pragma once

#include <memory>
#include <cuda_runtime.h>
#include <cuda/std/optional>
#include "ray/ray.h"
#include "scattering_record/scattering_record.h"


struct HitRecord {
	glm::vec3 p{};
	glm::vec3 normal{};
	float t;
	bool frontFace;

	__device__ void setFaceNormal(const Ray& ray, const glm::vec3& outwardNormal) {
		// outwardNormal is supposed to be normalized
		frontFace = glm::dot(ray.getDirection(), outwardNormal) < 0;
		normal = frontFace ? outwardNormal : outwardNormal * -1.0f;
		return;
	}
};


struct HitScatterRecord {
	cuda::std::optional<HitRecord> hitRec = cuda::std::nullopt;
	cuda::std::optional<ScatteringRecord> scatterRec = cuda::std::nullopt;
};