#pragma once
#include "hittable/hittable.h"
#include "linear_bvh_node.h"

class MeshBVH : public Hittable {
private:
    LinearBVHNode* nodes;
    glm::vec3* vertices;
    int* indices;
    Material* material;

    __device__ bool hitTriangle(const Ray& r, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, float& t, float& u, float& v_bary) const {
        const float EPSILON = 0.0000001f;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(r.getDirection(), edge2);
        float a = glm::dot(edge1, h);

        if (a < EPSILON) return false;

        float f = 1.0f / a;
        glm::vec3 s = r.getOrigin() - v0;
        u = f * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f) return false;

        glm::vec3 q = glm::cross(s, edge1);
        v_bary = f * glm::dot(r.getDirection(), q);
        if (v_bary < 0.0f || u + v_bary > 1.0f) return false;

        float tempT = f * glm::dot(edge2, q);
        if (tempT > EPSILON) {
            t = tempT;
            return true;
        }
        return false;
    }

public:
    __device__ MeshBVH(LinearBVHNode* nodes, glm::vec3* verts, int* indices, Material* mat)
        : nodes(nodes), vertices(verts), indices(indices), material(mat) {
    }

    __device__ virtual cuda::std::optional<Intersection> hit(const Ray& ray, float tMin, float tMax) const override {
        int stack[64];
        int stackPtr = 0;
        stack[stackPtr++] = 0;

        bool hitAnything = false;
        float closestT = tMax;

        float u, v;
        float bestU, bestV;
        int bestTriIndex = -1;

        while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
            LinearBVHNode node = nodes[nodeIdx];

            if (node.box.hit(ray, tMin, closestT)) {
                if (node.triCount > 0) {
                    for (int i = 0; i < node.triCount; i++) {
                        int baseIdx = node.triIndex + (i * 3);
                        int i0 = indices[baseIdx + 0];
                        int i1 = indices[baseIdx + 1];
                        int i2 = indices[baseIdx + 2];
                        glm::vec3 v0 = vertices[i0];
                        glm::vec3 v1 = vertices[i1];
                        glm::vec3 v2 = vertices[i2];
                        float t, u, v;
                        if (hitTriangle(ray, v0, v1, v2, t, u, v)) {
                            if (t < closestT && t > tMin) {
                                closestT = t;
                                hitAnything = true;
                                bestU = u;
                                bestV = v;
                                bestTriIndex = baseIdx;
                            }
                        }
                    }
                }
                else {
                    stack[stackPtr++] = node.rightIdx;
                    stack[stackPtr++] = node.leftIdx;
                }
            }
        }

        if (!hitAnything) return {};

        return constructIntersection(ray, closestT, bestTriIndex, bestU, bestV);
    }

    __device__ Intersection constructIntersection(const Ray& ray, float t, int triIdx, float u, float v) const {
        Intersection intersection{};
        HitRecord rec{};
        rec.t = t;
        rec.p = ray.At(t);

        int i0 = indices[triIdx + 0];
        int i1 = indices[triIdx + 1];
        int i2 = indices[triIdx + 2];

        glm::vec3 v0 = vertices[i0];
        glm::vec3 v1 = vertices[i1];
        glm::vec3 v2 = vertices[i2];
        glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        rec.setFaceNormal(ray, normal);
        intersection.hitRec = rec;
        intersection.mat = material;
        return intersection;
    }
    __device__ virtual Intersection constructIntersection(const Ray& ray, float t) const override { return {}; }
    __device__ virtual AABB boundingBox() const override {
        return nodes[0].box;
    }
};