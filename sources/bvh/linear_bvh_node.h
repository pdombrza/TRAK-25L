#pragma once
#include "aabb/aabb.h"

struct alignas(16) LinearBVHNode {
    AABB box;
    int leftIdx;      // Index of left child in the node array
    int rightIdx;     // Index of right child
    int triIndex;     // Index into the mesh's index buffer (start of triangle)
    int triCount;     // Number of triangles in this leaf (0 = internal node)
    int axis;         // Split axis (0=x, 1=y, 2=z) for traversal ordering
    int pad;
};