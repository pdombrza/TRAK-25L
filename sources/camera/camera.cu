#include "camera.h"

__device__ void Camera::initialize(int imgWidth, int imgHeight) {
	center = orientation.lookFrom;
	float theta = glm::radians(vFov);
	float h = tanf(theta / 2);
	float viewportHeight = 2 * h * focusDist;
	float viewportWidth = viewportHeight * ((float)imgWidth / imgHeight);

	w = glm::normalize(orientation.lookFrom - orientation.lookAt);
	u = glm::normalize(glm::cross(orientation.vUp, w));
	v = glm::cross(w, u);


	glm::vec3 viewportU = viewportWidth * u;
	glm::vec3 viewportV = viewportHeight * (-v);

	pixelDeltaU = viewportU / (float)imgWidth;
	pixelDeltaV = viewportV / (float)imgHeight;
	
	glm::vec3 viewportUpperLeft = center - (focusDist * w) - viewportU / 2.0f - viewportV / 2.0f;
	startPixelLoc = viewportUpperLeft + 0.5f * (pixelDeltaU + pixelDeltaV);

	// Camera defocus disk basis vectors calculations
	float defocusRadius = focusDist * tanf(glm::radians(defocusAngle / 2.0f));
	defocusDiskU = u * defocusRadius;
	defocusDiskV = v * defocusRadius;
}

__device__ Ray Camera::getRay(float u, float v, utils::random::RNG& rng) const {
	glm::vec3 offset = rng.sampleSquare();
	glm::vec3 pixelSample = startPixelLoc + (u + offset.x) * pixelDeltaU + (v + offset.y) * pixelDeltaV;
	glm::vec3 rayOrigin = (defocusAngle <= 0) ? center : defocusDiskSample(rng);
	glm::vec3 rayDirection = pixelSample - rayOrigin;
	return Ray(rayOrigin, rayDirection);
}

__device__ glm::vec3 Camera::defocusDiskSample(utils::random::RNG& rng) const {
	glm::vec3 p = rng.randomVec3InDisk();
	return center + p.x * defocusDiskU + p.y * defocusDiskV;
}