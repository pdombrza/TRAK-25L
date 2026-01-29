#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "utils/utils.h"

#define MAX_CONES 10

struct BoundingSphere {
    glm::vec3 center;
    float radius;

    __device__ BoundingSphere() : center(0.0f), radius(0.0f) {}
    __device__ BoundingSphere(const glm::vec3& c, float r) : center(c), radius(r) {}
};

// circular cone (eq 4) 
struct Cone {
    glm::vec3 apex;         // a - position of cone apex
    glm::vec3 direction;    // d - direction of cone axis
    float height;           // h - height of the cone
    float halfAngle;        // half angle of the cone
    bool valid;             // whether this cone is valid

    __device__ Cone() : apex(0.0f), direction(0.0f, 1.0f, 0.0f),
                       height(0.0f), halfAngle(0.0f), valid(false) {}

    __device__ Cone(const glm::vec3& surfacePoint, const BoundingSphere& sphere) {
        valid = true;
        apex = surfacePoint;

        float distToCenter = glm::length(surfacePoint - sphere.center);

        if (distToCenter < sphere.radius) {
            direction = glm::vec3(0.0f, 1.0f, 0.0f);
            height = distToCenter + sphere.radius;
            halfAngle = utils::PI / 2.0f;
        } else {
            direction = glm::normalize(sphere.center - surfacePoint);
            height = distToCenter + sphere.radius;
            halfAngle = asinf(sphere.radius / distToCenter);
        }
    }

    __device__ bool containsDirection(const glm::vec3& dir) const {
        if (!valid) return false;

        float cosAngle = glm::dot(glm::normalize(dir), direction);
        float angle = acosf(glm::clamp(cosAngle, -1.0f, 1.0f));
        return angle < halfAngle;
    }
};

struct ConeSet {
    Cone cones[MAX_CONES];
    int count;

    __device__ ConeSet() : count(0) {}

    __device__ void addCone(const Cone& cone) {
        if (count < MAX_CONES) {
            cones[count++] = cone;
        }
    }

    __device__ void clear() {
        count = 0;
    }
};

// Conical ray culling

__device__ inline bool isBehindSurface(const glm::vec3& p, const glm::vec3& n,
                                       const BoundingSphere& sphere) {
    return glm::dot(p - sphere.center, n) >= sphere.radius;
}

__device__ inline bool tryMergeCones(Cone& c1, const Cone& c2) {
    if (c1.halfAngle >= utils::PI / 2.0f) {
        c1.height = fmaxf(c1.height, c2.height);
        return true;
    }

    if (c2.halfAngle >= utils::PI / 2.0f) {
        c1.direction = c2.direction;
        c1.height = fmaxf(c1.height, c2.height);
        c1.halfAngle = c2.halfAngle;
        return true;
    }

    float dotProduct = glm::dot(c1.direction, c2.direction);
    float angle = acosf(glm::clamp(dotProduct, -1.0f, 1.0f));

    if (angle + c2.halfAngle < c1.halfAngle) {
        c1.height = fmaxf(c1.height, c2.height);
        return true;
    }

    if (angle + c1.halfAngle < c2.halfAngle) {
        c1.direction = c2.direction;
        c1.height = fmaxf(c1.height, c2.height);
        c1.halfAngle = c2.halfAngle;
        return true;
    }

    return false;
}

__device__ inline void constructAndMergeCones(const glm::vec3& p, const glm::vec3& n,
                                              BoundingSphere* spheres, int numSpheres,
                                              ConeSet& coneSet) {
    coneSet.clear();

    for (int i = 0; i < numSpheres; i++) {
        BoundingSphere& sphere = spheres[i];

        float distToCenter = glm::length(p - sphere.center);
        if (distToCenter >= sphere.radius && isBehindSurface(p, n, sphere)) {
            continue;
        }

        Cone cone(p, sphere);
        if (distToCenter < sphere.radius) {
            cone.direction = n;
        }

        bool merged = false;
        for (int j = 0; j < coneSet.count; j++) {
            if (tryMergeCones(coneSet.cones[j], cone)) {
                merged = true;
                break;
            }
        }

        if (!merged) {
            coneSet.addCone(cone);
        }
    }
}

__device__ inline bool shouldTraceRay(const glm::vec3& rayDir, const ConeSet& coneSet,
                                      float& maxRayLength) {
    if (coneSet.count == 0) {
        return false;
    }

    maxRayLength = 0.0f;
    bool insideCone = false;

    for (int i = 0; i < coneSet.count; i++) {
        const Cone& cone = coneSet.cones[i];
        if (cone.valid && cone.containsDirection(rayDir)) {
            insideCone = true;
            maxRayLength = fmaxf(maxRayLength, cone.height);
        }
    }

    return insideCone;
}
