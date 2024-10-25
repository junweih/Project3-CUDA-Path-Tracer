#include "JSONSceneLoader.h"
#include <iostream>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "scene.h"

#include <gltf-loader.h>

#define STBI_MSC_SECURE_CRT
#define TINYGLTF_IMPLEMENTATION

#include "tiny_gltf.h"

Scene* JSONSceneLoader::loadScene(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    transform(ext.begin(), ext.end(), ext.begin(), tolower);
    Scene* scene = new Scene();

    if (ext == "gltf") {
        loadGLTF(scene, filename);  // Pass filename to loadGLTF
    }
    else if (ext == "json") {
        json sceneData;
        file >> sceneData;
        loadMaterials(sceneData["materials"], scene);
        loadJsonObjects(sceneData["objects"], scene);
        loadCamera(sceneData["camera"], scene);
    }

    return scene;
}

void JSONSceneLoader::loadMaterials(const json& materials, Scene* scene) {
    for (const auto& material : materials) {
        Material newMaterial;
        newMaterial.color = glm::vec3(material["color"][0], material["color"][1], material["color"][2]);
        newMaterial.specular.exponent = material["specular"]["exponent"];
        newMaterial.specular.color = glm::vec3(material["specular"]["color"][0], material["specular"]["color"][1], material["specular"]["color"][2]);
        newMaterial.hasReflective = material["reflective"];
        newMaterial.hasRefractive = material["refractive"];
        newMaterial.indexOfRefraction = material["indexOfRefraction"];
        newMaterial.emittance = material["emittance"];
        newMaterial.hasTransmission = material["transmission"];
        scene->materials.push_back(newMaterial);
    }
}

void JSONSceneLoader::loadJsonObjects(const json& objects, Scene* scene) {
    for (const auto& object : objects) {
        Geom newGeom;
        newGeom.type = object["type"] == "sphere" ? SPHERE : CUBE;
        newGeom.materialid = object["materialId"];
        newGeom.translation = glm::vec3(object["translation"][0], object["translation"][1], object["translation"][2]);
        newGeom.rotation = glm::vec3(object["rotation"][0], object["rotation"][1], object["rotation"][2]);
        newGeom.scale = glm::vec3(object["scale"][0], object["scale"][1], object["scale"][2]);

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        scene->geoms.push_back(newGeom);
    }
}

void JSONSceneLoader::loadCamera(const json& camera, Scene* scene) {
    RenderState& state = scene->state;
    Camera& cam = state.camera;

    cam.resolution.x = camera["resolution"][0];
    cam.resolution.y = camera["resolution"][1];
    float fovy = camera["fovy"];
    state.iterations = camera["iterations"];
    state.traceDepth = camera["depth"];
    state.imageName = camera["outputFile"].get<std::string>();

    cam.position = glm::vec3(camera["eye"][0], camera["eye"][1], camera["eye"][2]);
    cam.lookAt = glm::vec3(camera["lookAt"][0], camera["lookAt"][1], camera["lookAt"][2]);
    cam.up = glm::vec3(camera["up"][0], camera["up"][1], camera["up"][2]);

    cam.focalLength = camera["focalLength"];
    cam.aperture = camera["aperture"];
    cam.dofEnabled = camera["dofEnabled"];

    // Calculate FOV and other camera properties
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * cam.resolution.x) / cam.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    cam.fov = glm::vec2(fovx, fovy);

    cam.right = glm::normalize(glm::cross(cam.view, cam.up));
    cam.pixelLength = glm::vec2(2 * xscaled / (float)cam.resolution.x,
        2 * yscaled / (float)cam.resolution.y);

    cam.view = glm::normalize(cam.lookAt - cam.position);

    int arraylen = cam.resolution.x * cam.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void JSONSceneLoader::loadGLTF(Scene* scene, const std::string& filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!warn.empty()) {
        std::cout << "GLTF Warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "GLTF Error: " << err << std::endl;
        return;
    }

    if (!ret) {
        std::cerr << "Failed to load GLTF file: " << filename << std::endl;
        return;
    }

    // Load Materials
    for (const auto& material : model.materials) {
        Material newMaterial;

        // Get base color
        if (material.pbrMetallicRoughness.baseColorFactor.size() >= 3) {
            newMaterial.color = glm::vec3(
                material.pbrMetallicRoughness.baseColorFactor[0],
                material.pbrMetallicRoughness.baseColorFactor[1],
                material.pbrMetallicRoughness.baseColorFactor[2]
            );
        }

        // Convert PBR properties to our material system
        float roughness = material.pbrMetallicRoughness.roughnessFactor;
        float metallic = material.pbrMetallicRoughness.metallicFactor;

        // Approximate specular from roughness
        newMaterial.specular.exponent = (1.0f - roughness) * 100.0f;
        newMaterial.specular.color = glm::vec3(1.0f);

        // Set reflective properties based on metallic
        newMaterial.hasReflective = metallic > 0.5f;

        // Add other properties with defaults
        newMaterial.hasRefractive = false;
        newMaterial.indexOfRefraction = 1.0f;
        newMaterial.emittance = 0.0f;
        newMaterial.hasTransmission = false;

        scene->materials.push_back(newMaterial);
    }

    // Load Meshes and Create Geometry
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            // Get indices
            const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

            // Get vertices
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

            // For each mesh primitive, create a cube approximation
            Geom newGeom;
            newGeom.type = CUBE; // Using cube as base primitive
            newGeom.materialid = primitive.material;

            // Calculate bounding box for scaling
            glm::vec3 min(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);
            glm::vec3 max(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);
            glm::vec3 center = (min + max) * 0.5f;
            glm::vec3 scale = (max - min) * 0.5f;

            newGeom.translation = center;
            newGeom.rotation = glm::vec3(0.0f);
            newGeom.scale = scale;

            // Build transformation matrices
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            scene->geoms.push_back(newGeom);
        }
    }

    // Set up a default camera if not specified
    if (model.cameras.empty()) {
        RenderState& state = scene->state;
        Camera& cam = state.camera;

        cam.resolution = glm::vec2(800, 600);
        cam.position = glm::vec3(0, 0, -10);
        cam.lookAt = glm::vec3(0, 0, 0);
        cam.up = glm::vec3(0, 1, 0);
        cam.fov = glm::vec2(45.0f);

        // Calculate other camera properties
        cam.view = glm::normalize(cam.lookAt - cam.position);
        cam.right = glm::normalize(glm::cross(cam.view, cam.up));

        // Setup render state
        state.iterations = 100;
        state.traceDepth = 5;
        state.imageName = "gltf_render.png";
    }
}
