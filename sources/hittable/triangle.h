#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "hittable.h"
#include "aabb/aabb.h"
#include "intersection/intersection.h"

class Triangle : public Hittable {
private:
    glm::vec3 v0, v1, v2;
    Material* material;
    glm::vec3 normal; 

public:
    __device__ Triangle(const glm::vec3& _v0, const glm::vec3& _v1, const glm::vec3& _v2, Material* mat)
        : v0(_v0), v1(_v1), v2(_v2), material(mat) {
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        normal = glm::normalize(glm::cross(edge1, edge2));
    }
    __device__ ~Triangle() {}
    __device__ virtual cuda::std::optional<Intersection> hit(const Ray& ray, float rayTMin, float rayTMax) const override {
        const float EPSILON = 0.0000001f;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(ray.getDirection(), edge2);
        float a = glm::dot(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return {};
        float f = 1.0f / a;
        glm::vec3 s = ray.getOrigin() - v0;
        float u = f * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f)
            return {};
        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(ray.getDirection(), q);
        if (v < 0.0f || u + v > 1.0f)
            return {};
        float t = f * glm::dot(edge2, q);
        if (t > rayTMin && t < rayTMax) {
            return constructIntersection(ray, t);
        }
        return {};
    }

    __device__ virtual Intersection constructIntersection(const Ray& ray, float t) const override {
        Intersection intersection{};
        HitRecord rec{};
        rec.t = t;
        rec.p = ray.At(t);

        rec.setFaceNormal(ray, normal);

        intersection.hitRec = rec;
        intersection.mat = material;
        return intersection;
    }

    __device__ virtual AABB boundingBox() const override {
        glm::vec3 minP(
            fminf(v0.x, fminf(v1.x, v2.x)),
            fminf(v0.y, fminf(v1.y, v2.y)),
            fminf(v0.z, fminf(v1.z, v2.z))
        );

        glm::vec3 maxP(
            fmaxf(v0.x, fmaxf(v1.x, v2.x)),
            fmaxf(v0.y, fmaxf(v1.y, v2.y)),
            fmaxf(v0.z, fmaxf(v1.z, v2.z))
        );

        const float epsilon = 0.0001f;
        if (fabsf(maxP.x - minP.x) < epsilon) { maxP.x += epsilon; minP.x -= epsilon; }
        if (fabsf(maxP.y - minP.y) < epsilon) { maxP.y += epsilon; minP.y -= epsilon; }
        if (fabsf(maxP.z - minP.z) < epsilon) { maxP.z += epsilon; minP.z -= epsilon; }

        return AABB(minP, maxP);
    }

    __device__ glm::vec3 getV0() const { return v0; }
    __device__ glm::vec3 getV1() const { return v1; }
    __device__ glm::vec3 getV2() const { return v2; }
};