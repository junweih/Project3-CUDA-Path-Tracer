#include "JSONSceneLoader.h"
#include <iostream>
#include <fstream>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "scene.h"

Scene* JSONSceneLoader::loadScene(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    json sceneData;
    file >> sceneData;

    Scene* scene = new Scene(); // Assuming Scene has a default constructor, or use appropriate parameters
    loadMaterials(sceneData["materials"], scene);
    loadObjects(sceneData["objects"], scene);
    loadCamera(sceneData["camera"], scene);

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

void JSONSceneLoader::loadObjects(const json& objects, Scene* scene) {
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
    state.imageName = camera["outputFile"];

    cam.position = glm::vec3(camera["eye"][0], camera["eye"][1], camera["eye"][2]);
    cam.lookAt = glm::vec3(camera["lookAt"][0], camera["lookAt"][1], camera["lookAt"][2]);
    cam.up = glm::vec3(camera["up"][0], camera["up"][1], camera["up"][2]);

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