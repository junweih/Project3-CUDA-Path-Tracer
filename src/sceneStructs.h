#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    GLTF_MESH  // New type for GLTF meshes
};

/**
 * Triangle intersection structure to store mesh data
 */
struct Triangle {
    glm::vec3 v0, v1, v2;    // Vertices
    glm::vec3 n0, n1, n2;    // Vertex normals
    glm::vec2 t0, t1, t2;    // Texture coordinates
};

// Add new struct for storing mesh data
struct MeshData {
    Triangle* dev_triangles;  // Device pointer to triangle array
    int numTriangles;
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // Add mesh data pointer (only valid for GLTF_MESH type)
    MeshData* meshData;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    float hasTransmission;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float focalLength;
    float aperture;
    bool dofEnabled;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    glm::vec3 throughput;   // represents how materials we have encountered so far will alter the color of a light source when it scttaers off of them
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
