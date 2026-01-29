#pragma once
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
#include "linear_bvh_node.h"

struct BVHPrimitive {
    int triIndex;
    AABB box;
    glm::vec3 center;
};

class BVHBuilder {
public:
    std::vector<LinearBVHNode> nodes;
    std::vector<int> orderedIndices;

    void build(const std::vector<glm::vec3>& vertices, const std::vector<int>& indices) {
        nodes.clear();
        orderedIndices.clear();
        std::vector<BVHPrimitive> primitives;
        int numTriangles = indices.size() / 3;

        for (int i = 0; i < numTriangles; i++) {
            glm::vec3 v0 = vertices[indices[i * 3 + 0]];
            glm::vec3 v1 = vertices[indices[i * 3 + 1]];
            glm::vec3 v2 = vertices[indices[i * 3 + 2]];

            glm::vec3 minP = glm::min(v0, glm::min(v1, v2));
            glm::vec3 maxP = glm::max(v0, glm::max(v1, v2));
            const float eps = 0.0001f;
            if (abs(maxP.x - minP.x) < eps) { minP.x -= eps; maxP.x += eps; }
            if (abs(maxP.y - minP.y) < eps) { minP.y -= eps; maxP.y += eps; }
            if (abs(maxP.z - minP.z) < eps) { minP.z -= eps; maxP.z += eps; }

            BVHPrimitive p;
            p.triIndex = i * 3;
            p.box = AABB(minP, maxP);
            p.center = (minP + maxP) * 0.5f;
            primitives.push_back(p);
        }
        int rootIdx = buildRecursive(primitives, 0, primitives.size());
    }

private:
    int buildRecursive(std::vector<BVHPrimitive>& prims, int start, int end) {
        LinearBVHNode node;
        node.leftIdx = -1;
        node.rightIdx = -1;
        node.triCount = 0;

        AABB totalBox = prims[start].box;
        for (int i = start + 1; i < end; i++) {
            totalBox = joinAABBs(totalBox, prims[i].box);
        }
        node.box = totalBox;

        int count = end - start;
        if (count <= 2) {
            node.triIndex = orderedIndices.size();
            node.triCount = count;
            for (int i = start; i < end; i++) {
                orderedIndices.push_back(prims[i].triIndex);
            }
            nodes.push_back(node);
            return nodes.size() - 1;
        }
        int axis = totalBox.longestAxis();
        node.axis = axis;

        int mid = (start + end) / 2;
        auto comparator = [axis](const BVHPrimitive& a, const BVHPrimitive& b) {
            return a.center[axis] < b.center[axis];
            };
        std::nth_element(prims.begin() + start, prims.begin() + mid, prims.begin() + end, comparator);
        int currentNodeIdx = nodes.size();
        nodes.push_back(node);
        int leftChild = buildRecursive(prims, start, mid);
        int rightChild = buildRecursive(prims, mid, end);
        nodes[currentNodeIdx].leftIdx = leftChild;
        nodes[currentNodeIdx].rightIdx = rightChild;

        return currentNodeIdx;
    }
};