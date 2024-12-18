﻿#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = glm::min(t1, t2);
        outside = true;
    } else {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

/**
 * Tests ray intersection with a triangle using Möller–Trumbore algorithm
 */
__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& texCoord,
    bool& outside) {
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\nTriangle test input:\n");
        printf("Triangle vertices:\n");
        printf("v0: (%f,%f,%f)\n", tri.v0.x, tri.v0.y, tri.v0.z);
        printf("v1: (%f,%f,%f)\n", tri.v1.x, tri.v1.y, tri.v1.z);
        printf("v2: (%f,%f,%f)\n", tri.v2.x, tri.v2.y, tri.v2.z);
        printf("Triangle normals:\n");
        printf("n0: (%f,%f,%f)\n", tri.n0.x, tri.n0.y, tri.n0.z);
        printf("n1: (%f,%f,%f)\n", tri.n1.x, tri.n1.y, tri.n1.z);
        printf("n2: (%f,%f,%f)\n", tri.n2.x, tri.n2.y, tri.n2.z);
        printf("Ray:\n");
        printf("origin: (%f,%f,%f), direction: (%f,%f,%f)\n",
            r.origin.x, r.origin.y, r.origin.z,
            r.direction.x, r.direction.y, r.direction.z);
}
#endif

    // Edge vectors
    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;

    // Calculate determinant
    glm::vec3 h = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, h);

#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Edge1: (%f,%f,%f)\n", edge1.x, edge1.y, edge1.z);
        printf("Edge2: (%f,%f,%f)\n", edge2.x, edge2.y, edge2.z);
        printf("P (ray direction x edge2): (%f,%f,%f)\n", h.x, h.y, h.z);
        printf("Determinant: %f\n", det);
    }
#endif

    // Check if ray is parallel to triangle
    if (det > -EPSILON && det < EPSILON) {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("REJECTED: Ray parallel to triangle (det near zero)\n");
        }
#endif
        return -1.0f;
    }

    float invDet = 1.0f / det;

    // Calculate u parameter
    glm::vec3 s = r.origin - tri.v0;
    float u = invDet * glm::dot(s, h);

#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("T vector (origin - v0): (%f,%f,%f)\n", s.x, s.y, s.z);
        printf("u parameter: %f\n", u);
    }
#endif

    // Check if intersection is outside triangle
    if (u < 0.0f || u > 1.0f) {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("REJECTED: u parameter out of range [0,1]: %f\n", u);
        }
#endif
        return -1.0f;
    }

    // Calculate v parameter
    glm::vec3 q = glm::cross(s, edge1);
    float v = invDet * glm::dot(r.direction, q);

#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Q vector (T x edge1): (%f,%f,%f)\n", q.x, q.y, q.z);
        printf("v parameter: %f\n", v);
    }
#endif

    // Check if intersection is outside triangle
    if (v < 0.0f || u + v > 1.0f) {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("REJECTED: v parameter out of range [0,1] or u+v > 1: v=%f, u+v=%f\n", v, u + v);
        }
#endif
        return -1.0f;
    }

    // Calculate t
    float t = invDet * glm::dot(edge2, q);

#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("t parameter (distance): %f\n", t);
    }
#endif

    // Check if intersection is behind ray origin
    if (t <= EPSILON) {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("REJECTED: Intersection behind ray origin (t <= EPSILON): %f\n", t);
        }
#endif
        return -1.0f;
    }

    // Compute intersection point
    intersectionPoint = getPointOnRay(r, t);

    // Interpolate normal using barycentric coordinates
    float w = 1.0f - u - v;
    normal = glm::normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);

    // Interpolate texture coordinates
    texCoord = w * tri.t0 + u * tri.t1 + v * tri.t2;

    // Ray always hits triangle from outside
    outside = true;

#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("HIT FOUND!\n");
        printf("Distance (t): %f\n", t);
        printf("Barycentric coordinates (w,u,v): (%f,%f,%f)\n", w, u, v);
        printf("Hit point: (%f,%f,%f)\n", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z);
        printf("Interpolated normal: (%f,%f,%f)\n", normal.x, normal.y, normal.z);
        printf("Texture coordinates: (%f,%f)\n", texCoord.x, texCoord.y);
        printf("-----------------------------------------\n");
    }
#endif

    return t;
}

/**
 * Tests intersection with a transformed triangle mesh
 */
__host__ __device__ float meshIntersectionTest(
    const Triangle* triangles,
    const int numTriangles,
    const Geom& mesh,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& texCoord,
    bool& outside) {

    // Debug print for input ray
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("World space ray - Origin: (%f,%f,%f), Direction: (%f,%f,%f)\n",
            r.origin.x, r.origin.y, r.origin.z,
            r.direction.x, r.direction.y, r.direction.z);

        // Print transform matrix
        printf("Transform matrix:\n");
        for (int i = 0; i < 4; i++) {
            printf("%f %f %f %f\n",
                mesh.transform[i][0], mesh.transform[i][1],
                mesh.transform[i][2], mesh.transform[i][3]);
        }

        printf("Inverse transform matrix:\n");
        for (int i = 0; i < 4; i++) {
            printf("%f %f %f %f\n",
                mesh.inverseTransform[i][0], mesh.inverseTransform[i][1],
                mesh.inverseTransform[i][2], mesh.inverseTransform[i][3]);
        }
    }
#endif

    // Transform ray to object space
    Ray localRay;
    localRay.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    localRay.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float closest_t = FLT_MAX;
    glm::vec3 temp_point, temp_normal;
    glm::vec2 temp_texcoord;
    bool temp_outside;

    // Test intersection with all triangles
    for (int i = 0; i < numTriangles; i++) {
        float t = triangleIntersectionTest(
            triangles[i],
            localRay,
            temp_point,
            temp_normal,
            temp_texcoord,
            temp_outside
        );

        if (t > 0.0f && t < closest_t) {
            closest_t = t;

            // Transform intersection point back to world space
            intersectionPoint = multiplyMV(mesh.transform, glm::vec4(temp_point, 1.0f));

            // Transform normal back to world space
            normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(temp_normal, 0.0f)));

            texCoord = temp_texcoord;
            outside = temp_outside;
        }
    }

    if (closest_t == FLT_MAX) return -1.0f;

#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Final result - t: %f, Point: (%f,%f,%f), Normal: (%f,%f,%f)\n",
            closest_t,
            intersectionPoint.x, intersectionPoint.y, intersectionPoint.z,
            normal.x, normal.y, normal.z);
    }
#endif

    return glm::length(r.origin - intersectionPoint);
}
/**
 * Structure to store GLTF mesh data
 */
struct GltfMesh {
    Triangle* triangles;      // Array of triangles
    int numTriangles;        // Number of triangles in the mesh
    Material material;       // Material for the mesh
};