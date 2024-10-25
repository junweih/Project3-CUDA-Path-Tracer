#include <iostream>
#include "scene.h"
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <algorithm>

#define STBI_MSC_SECURE_CRT
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

Scene::Scene() {}

Scene::Scene(const std::string& filename) {
    loadFromFile(filename);
}

void Scene::loadFromFile(const std::string& filename) {
    std::cout << "Reading scene from " << filename << "..." << std::endl;

    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == "gltf") {
        loadGLTF(filename);
    }
    else if (ext == "json") {
        loadJSON(filename);
    }
    else {
        throw std::runtime_error("Unsupported file format: " + ext);
    }
}

void Scene::loadJSON(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    json sceneData;
    file >> sceneData;

    loadMaterials(sceneData["materials"]);
    loadGeoms(sceneData["objects"]);
    loadCamera(sceneData["camera"]);
}

void Scene::loadMaterials(const json& jsonMaterials) {
    for (const auto& material : jsonMaterials) {
        Material newMaterial;
        newMaterial.color = glm::vec3(
            material["color"][0],
            material["color"][1],
            material["color"][2]
        );
        newMaterial.specular.exponent = material["specular"]["exponent"];
        newMaterial.specular.color = glm::vec3(
            material["specular"]["color"][0],
            material["specular"]["color"][1],
            material["specular"]["color"][2]
        );
        newMaterial.hasReflective = material["reflective"];
        newMaterial.hasRefractive = material["refractive"];
        newMaterial.indexOfRefraction = material["indexOfRefraction"];
        newMaterial.emittance = material["emittance"];
        newMaterial.hasTransmission = material["transmission"];

        materials.push_back(newMaterial);
    }
}

void Scene::loadGeoms(const json& objects) {
    for (const auto& object : objects) {
        Geom newGeom;
        newGeom.type = object["type"] == "sphere" ? SPHERE : CUBE;
        newGeom.materialid = object["materialId"];

        newGeom.translation = glm::vec3(
            object["translation"][0],
            object["translation"][1],
            object["translation"][2]
        );
        newGeom.rotation = glm::vec3(
            object["rotation"][0],
            object["rotation"][1],
            object["rotation"][2]
        );
        newGeom.scale = glm::vec3(
            object["scale"][0],
            object["scale"][1],
            object["scale"][2]
        );

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
}

void Scene::loadCamera(const json& camera) {
    RenderState& state = this->state;
    Camera& cam = state.camera;

    cam.resolution.x = camera["resolution"][0];
    cam.resolution.y = camera["resolution"][1];
    float fovy = camera["fovy"];
    state.iterations = camera["iterations"];
    state.traceDepth = camera["depth"];
    state.imageName = camera["outputFile"].get<std::string>();

    cam.position = glm::vec3(
        camera["eye"][0],
        camera["eye"][1],
        camera["eye"][2]
    );
    cam.lookAt = glm::vec3(
        camera["lookAt"][0],
        camera["lookAt"][1],
        camera["lookAt"][2]
    );
    cam.up = glm::vec3(
        camera["up"][0],
        camera["up"][1],
        camera["up"][2]
    );

    cam.focalLength = camera["focalLength"];
    cam.aperture = camera["aperture"];
    cam.dofEnabled = camera["dofEnabled"];

    // Calculate FOV and other camera properties
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * cam.resolution.x) / cam.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    cam.fov = glm::vec2(fovx, fovy);

    cam.view = glm::normalize(cam.lookAt - cam.position);
    cam.right = glm::normalize(glm::cross(cam.view, cam.up));
    cam.pixelLength = glm::vec2(
        2 * xscaled / (float)cam.resolution.x,
        2 * yscaled / (float)cam.resolution.y
    );

    // Initialize image buffer
    int arraylen = cam.resolution.x * cam.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadGLTF(const std::string& filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        std::cout << "GLTF Warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "GLTF Error: " << err << std::endl;
        return;
    }
    if (!ret) {
        throw std::runtime_error("Failed to load GLTF file: " + filename);
    }

    // Load Materials
    for (const auto& material : model.materials) {
        Material newMaterial;

        if (material.pbrMetallicRoughness.baseColorFactor.size() >= 3) {
            newMaterial.color = glm::vec3(
                material.pbrMetallicRoughness.baseColorFactor[0],
                material.pbrMetallicRoughness.baseColorFactor[1],
                material.pbrMetallicRoughness.baseColorFactor[2]
            );
        }

        float roughness = material.pbrMetallicRoughness.roughnessFactor;
        float metallic = material.pbrMetallicRoughness.metallicFactor;

        newMaterial.specular.exponent = (1.0f - roughness) * 100.0f;
        newMaterial.specular.color = glm::vec3(1.0f);
        newMaterial.hasReflective = metallic > 0.5f;
        newMaterial.hasRefractive = false;
        newMaterial.indexOfRefraction = 1.0f;
        newMaterial.emittance = 0.0f;
        newMaterial.hasTransmission = false;

        materials.push_back(newMaterial);
    }

    // Load Meshes
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            const tinygltf::Accessor& posAccessor =
                model.accessors[primitive.attributes.find("POSITION")->second];

            Geom newGeom;
            newGeom.type = CUBE;  // Using cube as base primitive
            newGeom.materialid = primitive.material;

            // Calculate bounding box
            glm::vec3 min(
                posAccessor.minValues[0],
                posAccessor.minValues[1],
                posAccessor.minValues[2]
            );
            glm::vec3 max(
                posAccessor.maxValues[0],
                posAccessor.maxValues[1],
                posAccessor.maxValues[2]
            );

            glm::vec3 center = (min + max) * 0.5f;
            glm::vec3 scale = (max - min) * 0.5f;

            newGeom.translation = center;
            newGeom.rotation = glm::vec3(0.0f);
            newGeom.scale = scale;

            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }

    // Set up default camera if none specified
    if (model.cameras.empty()) {
        setupDefaultCamera();
    }
}

void Scene::setupDefaultCamera() {
    Camera& cam = state.camera;

    cam.resolution = glm::vec2(800, 600);
    cam.position = glm::vec3(0, 0, -10);
    cam.lookAt = glm::vec3(0, 0, 0);
    cam.up = glm::vec3(0, 1, 0);
    cam.fov = glm::vec2(45.0f);

    cam.view = glm::normalize(cam.lookAt - cam.position);
    cam.right = glm::normalize(glm::cross(cam.view, cam.up));

    state.iterations = 100;
    state.traceDepth = 5;
    state.imageName = "gltf_render.png";

    // Initialize image buffer
    int arraylen = cam.resolution.x * cam.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}