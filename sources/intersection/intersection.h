#pragma once
#include <cuda_runtime.h>

#include "hitrec/hitrec.h"
#include "material/material.h"

struct Intersection {
	HitRecord hitRec;
	Material* mat;
};
