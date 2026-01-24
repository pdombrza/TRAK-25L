#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "utils/utils.h"

// Maximum number of cones per pixel
#define MAX_CONES 10

// Structure representing a bounding sphere
struct BoundingSphere {
    glm::vec3 center;
    float radius;

    __device__ BoundingSphere() : center(0.0f), radius(0.0f) {}
    __device__ BoundingSphere(const glm::vec3& c, float r) : center(c), radius(r) {}
};

// Structure representing a circular cone (from Equation 4 in the paper)
struct Cone {
    glm::vec3 apex;         // a - position of cone apex
    glm::vec3 direction;    // d - direction of cone axis
    float height;           // h - height of the cone
    float halfAngle;        // ψ - half angle of the cone
    bool valid;             // whether this cone is valid

    __device__ Cone() : apex(0.0f), direction(0.0f, 1.0f, 0.0f),
                       height(0.0f), halfAngle(0.0f), valid(false) {}

    // Construct cone from bounding sphere (Equation 4)
    __device__ Cone(const glm::vec3& surfacePoint, const BoundingSphere& sphere) {
        valid = true;
        apex = surfacePoint;

        float distToCenter = glm::length(surfacePoint - sphere.center);

        // Check if point is inside sphere
        if (distToCenter < sphere.radius) {
            // Point is inside the bounding sphere - cone covers hemisphere
            direction = glm::vec3(0.0f, 1.0f, 0.0f); // Will be set to surface normal
            height = distToCenter + sphere.radius;
            halfAngle = utils::PI / 2.0f;
        } else {
            direction = glm::normalize(sphere.center - surfacePoint);
            height = distToCenter + sphere.radius;
            halfAngle = asinf(sphere.radius / distToCenter);
        }
    }

    // Check if a direction is inside this cone
    __device__ bool containsDirection(const glm::vec3& dir) const {
        if (!valid) return false;

        float cosAngle = glm::dot(glm::normalize(dir), direction);
        float angle = acosf(glm::clamp(cosAngle, -1.0f, 1.0f));
        return angle < halfAngle;
    }
};

// Structure to hold all cones for a surface point
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

// Conical ray culling functions

// Check if bounding sphere is behind the surface (Figure 5)
__device__ inline bool isBehindSurface(const glm::vec3& p, const glm::vec3& n,
                                       const BoundingSphere& sphere) {
    return glm::dot(p - sphere.center, n) >= sphere.radius;
}

// Merge two cones if possible (Equation 5 and Figure 6)
__device__ inline bool tryMergeCones(Cone& c1, const Cone& c2) {
    // Check if c1 can contain c2
    if (c1.halfAngle >= utils::PI / 2.0f) {
        // c1 is already hemisphere
        c1.height = fmaxf(c1.height, c2.height);
        return true;
    }

    if (c2.halfAngle >= utils::PI / 2.0f) {
        // c2 is hemisphere, replace c1
        c1.direction = c2.direction;
        c1.height = fmaxf(c1.height, c2.height);
        c1.halfAngle = c2.halfAngle;
        return true;
    }

    float dotProduct = glm::dot(c1.direction, c2.direction);
    float angle = acosf(glm::clamp(dotProduct, -1.0f, 1.0f));

    // Check if c2 is inside c1 (Figure 6 left)
    if (angle + c2.halfAngle < c1.halfAngle) {
        c1.height = fmaxf(c1.height, c2.height);
        return true;
    }

    // Check if c1 can be expanded to contain c2 (Figure 6 middle)
    if (angle + c1.halfAngle < c2.halfAngle) {
        c1.direction = c2.direction;
        c1.height = fmaxf(c1.height, c2.height);
        c1.halfAngle = c2.halfAngle;
        return true;
    }

    // Cannot merge (Figure 6 right)
    return false;
}

// Algorithm 1: Construct and merge cones
__device__ inline void constructAndMergeCones(const glm::vec3& p, const glm::vec3& n,
                                              BoundingSphere* spheres, int numSpheres,
                                              ConeSet& coneSet) {
    coneSet.clear();

    for (int i = 0; i < numSpheres; i++) {
        BoundingSphere& sphere = spheres[i];

        float distToCenter = glm::length(p - sphere.center);

        // Ignore spheres behind the surface (line 4 in Algorithm 1)
        if (distToCenter >= sphere.radius && isBehindSurface(p, n, sphere)) {
            continue;
        }

        // Construct cone from bounding sphere
        Cone cone(p, sphere);

        // Special case: point inside sphere (lines 6-8)
        if (distToCenter < sphere.radius) {
            cone.direction = n;
        }

        // Try to merge with existing cones (lines 13-30)
        bool merged = false;
        for (int j = 0; j < coneSet.count; j++) {
            if (tryMergeCones(coneSet.cones[j], cone)) {
                merged = true;
                break;
            }
        }

        // If not merged, add as new cone (lines 27-29)
        if (!merged) {
            coneSet.addCone(cone);
        }
    }
}

// Algorithm 2: Check if a ray should be traced (conical ray culling)
__device__ inline bool shouldTraceRay(const glm::vec3& rayDir, const ConeSet& coneSet,
                                      float& maxRayLength) {
    if (coneSet.count == 0) {
        return false; // No dynamic objects, no need to trace
    }

    maxRayLength = 0.0f;
    bool insideCone = false;

    // Check if ray is inside any cone (lines 6-11 in Algorithm 2)
    for (int i = 0; i < coneSet.count; i++) {
        const Cone& cone = coneSet.cones[i];
        if (cone.valid && cone.containsDirection(rayDir)) {
            insideCone = true;
            maxRayLength = fmaxf(maxRayLength, cone.height);
        }
    }

    return insideCone;
}
