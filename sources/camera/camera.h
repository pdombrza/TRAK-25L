#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "hittable/hittable.h"
#include "hitrec/hitrec.h"
#include "ray/ray.h"
#include "utils/utils.h"


struct CameraOrientation {
	glm::vec3 lookFrom{};
	glm::vec3 lookAt{ 0.0f, 0.0f, -1.0f };
	glm::vec3 vUp{ 0.0f, 1.0f, 0.0f }; // view up vector
};


class Camera {
private:
	float aspectRatio = 16.0f / 9.0f;
	float vFov = 90.0f; // in degrees
	float defocusAngle = 0.0f; // Angle of the cone from pov to defocus disk
	float focusDist = 10.0f; // The new focal length
	glm::vec3 center{};
	glm::vec3 startPixelLoc{};
	glm::vec3 pixelDeltaU{};
	glm::vec3 pixelDeltaV{};
	glm::vec3 u{}, v{}, w{};
	glm::vec3 defocusDiskU{};
	glm::vec3 defocusDiskV{};
	CameraOrientation orientation{};
public:
	__host__ __device__ Camera() {};
	__host__ __device__ Camera(CameraOrientation orientation) : orientation(orientation) {};
	__host__ __device__ Camera(float aspectRatio) : aspectRatio(aspectRatio) {};
	//Camera(float vFov) : vFov(vFov) {};
	__host__ __device__ Camera(float vFov, float aspectRatio) : vFov(vFov), aspectRatio(aspectRatio) {};
	__host__ __device__ Camera(CameraOrientation orientation, float vFov, float aspectRatio) : orientation(orientation), vFov(vFov), aspectRatio(aspectRatio) {};
	__host__ __device__ Camera(float aspectRatio, float focalLength, float viewportHeight, const glm::vec3& center) : aspectRatio(aspectRatio), center(center) {};
	__host__ __device__ ~Camera() = default;
	__device__ void initialize(int imgWidth, int imgHeight);
	__device__ glm::vec3 defocusDiskSample(utils::random::RNG& rng) const;
	__device__ Ray getRay(float h, float w, utils::random::RNG& rng) const;
	__device__ void setAspectRatio(float newAspectRatio) { aspectRatio = newAspectRatio; };
	__device__ float getAspectRatio() const { return aspectRatio; };
	__device__ void setCenter(glm::vec3& newCameraCenter) { center = newCameraCenter; };
	__device__ float getVFov() const { return vFov; };
	__device__ void setVFov(float newVFov) { vFov = newVFov; };
	__device__ CameraOrientation getCameraOrientation() const { return orientation; };
	__device__ void setCameraOrientation(CameraOrientation& newOrientation) { orientation = newOrientation; };
	__device__ glm::vec3 getCenter() const { return center; };
	__device__ glm::vec3 getStartPixelLoc() const { return startPixelLoc; };
	__device__ glm::vec3 getPixelDeltaU() const { return pixelDeltaU; };
	__device__ glm::vec3 getPixelDeltaV() const { return pixelDeltaV; };
	__device__ float getDefocusAngle() const { return defocusAngle; };
	__device__ void setDefocusAngle(float newDefocusAngle) { defocusAngle = newDefocusAngle; };
	__device__ float getFocusDist() const { return focusDist; };
	__device__ void setFocusDist(float newFocusDist) { focusDist = newFocusDist; };
};