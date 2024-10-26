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

Scene::~Scene() {
    for (auto& geom : geoms) {
        if (geom.type == GLTF_MESH && geom.meshData != nullptr) {
            cudaFree(geom.meshData->dev_triangles);
            delete geom.meshData;
        }
    }
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

void loadMeshFromPrimitive(Geom& geom, const tinygltf::Model& model, const tinygltf::Primitive& primitive) {
    // Get vertex position data
    const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
    const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
    const float* positions = reinterpret_cast<const float*>(&(model.buffers[posView.buffer].data[posView.byteOffset + posAccessor.byteOffset]));

    // Get normal data
    const tinygltf::Accessor& normalAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
    const tinygltf::BufferView& normalView = model.bufferViews[normalAccessor.bufferView];
    const float* normals = reinterpret_cast<const float*>(&(model.buffers[normalView.buffer].data[normalView.byteOffset + normalAccessor.byteOffset]));

    // Get UV data
    const tinygltf::Accessor& uvAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
    const tinygltf::BufferView& uvView = model.bufferViews[uvAccessor.bufferView];
    const float* uvs = reinterpret_cast<const float*>(&(model.buffers[uvView.buffer].data[uvView.byteOffset + uvAccessor.byteOffset]));

    // Get indices
    const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
    const tinygltf::BufferView& indexView = model.bufferViews[indexAccessor.bufferView];
    const uint16_t* indices = reinterpret_cast<const uint16_t*>(&(model.buffers[indexView.buffer].data[indexView.byteOffset + indexAccessor.byteOffset]));

    // Create triangles
    int numTriangles = indexAccessor.count / 3;
    std::vector<Triangle> triangles(numTriangles);

    for (int i = 0; i < numTriangles; i++) {
        uint16_t idx0 = indices[i * 3];
        uint16_t idx1 = indices[i * 3 + 1];
        uint16_t idx2 = indices[i * 3 + 2];

        // Set vertices
        triangles[i].v0 = glm::vec3(
            positions[idx0 * 3],
            positions[idx0 * 3 + 1],
            positions[idx0 * 3 + 2]
        );
        triangles[i].v1 = glm::vec3(
            positions[idx1 * 3],
            positions[idx1 * 3 + 1],
            positions[idx1 * 3 + 2]
        );
        triangles[i].v2 = glm::vec3(
            positions[idx2 * 3],
            positions[idx2 * 3 + 1],
            positions[idx2 * 3 + 2]
        );

        // Set normals
        triangles[i].n0 = glm::vec3(
            normals[idx0 * 3],
            normals[idx0 * 3 + 1],
            normals[idx0 * 3 + 2]
        );
        triangles[i].n1 = glm::vec3(
            normals[idx1 * 3],
            normals[idx1 * 3 + 1],
            normals[idx1 * 3 + 2]
        );
        triangles[i].n2 = glm::vec3(
            normals[idx2 * 3],
            normals[idx2 * 3 + 1],
            normals[idx2 * 3 + 2]
        );

        // Set UVs
        triangles[i].t0 = glm::vec2(
            uvs[idx0 * 2],
            uvs[idx0 * 2 + 1]
        );
        triangles[i].t1 = glm::vec2(
            uvs[idx1 * 2],
            uvs[idx1 * 2 + 1]
        );
        triangles[i].t2 = glm::vec2(
            uvs[idx2 * 2],
            uvs[idx2 * 2 + 1]
        );
    }

    // Allocate and set up mesh data
    geom.meshData = new MeshData();
    geom.meshData->numTriangles = numTriangles;

    // Allocate GPU memory and copy triangle data
    cudaMalloc(&geom.meshData->dev_triangles, numTriangles * sizeof(Triangle));
    cudaMemcpy(geom.meshData->dev_triangles, triangles.data(), numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
}

void Scene::loadGLTF(const std::string& filename) {
    std::cout << "\n=== Loading GLTF Scene: " << filename << " ===\n" << std::endl;

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

    // Print scene information
    std::cout << "Scene Information:" << std::endl;
    std::cout << "- Scenes: " << model.scenes.size() << std::endl;
    std::cout << "- Nodes: " << model.nodes.size() << std::endl;
    std::cout << "- Meshes: " << model.meshes.size() << std::endl;
    std::cout << "- Materials: " << model.materials.size() << std::endl;
    std::cout << "- Textures: " << model.textures.size() << std::endl;
    std::cout << "- Images: " << model.images.size() << std::endl;
    std::cout << "- Cameras: " << model.cameras.size() << std::endl;
    std::cout << std::endl;

    // Load Materials
    std::cout << "Loading Materials..." << std::endl;
    for (size_t i = 0; i < model.materials.size(); i++) {
        const auto& material = model.materials[i];
        Material newMaterial;

        std::cout << "Material " << i << ": " << (material.name.empty() ? "unnamed" : material.name) << std::endl;

        // Set default color if no texture
        newMaterial.color = glm::vec3(1.0f);
        if (material.pbrMetallicRoughness.baseColorFactor.size() >= 3) {
            newMaterial.color = glm::vec3(
                material.pbrMetallicRoughness.baseColorFactor[0],
                material.pbrMetallicRoughness.baseColorFactor[1],
                material.pbrMetallicRoughness.baseColorFactor[2]
            );
            std::cout << "  - Base Color: ("
                << newMaterial.color.x << ", "
                << newMaterial.color.y << ", "
                << newMaterial.color.z << ")" << std::endl;
        }

        // Basic PBR conversion
        float roughness = material.pbrMetallicRoughness.roughnessFactor;
        float metallic = material.pbrMetallicRoughness.metallicFactor;

        newMaterial.specular.exponent = (1.0f - roughness) * 100.0f;
        newMaterial.specular.color = glm::vec3(metallic);
        newMaterial.hasReflective = metallic > 0.5f ? 1.0f : 0.0f;

        std::cout << "  - Roughness: " << roughness << std::endl;
        std::cout << "  - Metallic: " << metallic << std::endl;
        std::cout << "  - Specular Exponent: " << newMaterial.specular.exponent << std::endl;
        std::cout << "  - Is Reflective: " << (newMaterial.hasReflective ? "yes" : "no") << std::endl;

        // Set defaults for other properties
        newMaterial.hasRefractive = 0.0f;
        newMaterial.indexOfRefraction = 1.0f;
        newMaterial.emittance = 0.0f;
        newMaterial.hasTransmission = 0.0f;

        materials.push_back(newMaterial);
    }
    std::cout << std::endl;

    // Load Meshes
    std::cout << "Loading Meshes..." << std::endl;
    size_t totalPrimitives = 0;
    size_t totalVertices = 0;
    size_t totalIndices = 0;

    for (size_t i = 0; i < model.meshes.size(); i++) {
        const auto& mesh = model.meshes[i];
        std::cout << "Mesh " << i << ": " << (mesh.name.empty() ? "unnamed" : mesh.name) << std::endl;
        std::cout << "  - Primitives: " << mesh.primitives.size() << std::endl;

        for (const auto& primitive : mesh.primitives) {
            totalPrimitives++;

            // Get position accessor
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
            totalVertices += posAccessor.count;

            // Get index accessor
            const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
            totalIndices += indexAccessor.count;

            Geom newGeom;
            newGeom.type = GLTF_MESH;
            newGeom.materialid = primitive.material;

            try {
                loadMeshFromPrimitive(newGeom, model, primitive);
                std::cout << "    - Loaded " << newGeom.meshData->numTriangles << " triangles" << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading mesh primitive: " << e.what() << std::endl;
                continue;
            }

            // Set default transforms
            newGeom.translation = glm::vec3(0.0f);
            newGeom.rotation = glm::vec3(0.0f);
            newGeom.scale = glm::vec3(1.0f);

            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            std::cout << "    - Material: " << newGeom.materialid << std::endl;
            std::cout << "    - Vertices: " << posAccessor.count << std::endl;
            std::cout << "    - Indices: " << indexAccessor.count << std::endl;

            geoms.push_back(newGeom);
        }
    }

    // After loading mesh data, add these debug prints
    for (size_t i = 0; i < model.meshes.size(); i++) {
        const auto& mesh = model.meshes[i];
        std::cout << "\nMesh " << i << " debug:" << std::endl;

        for (const auto& primitive : mesh.primitives) {
            // Get vertex position data
            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
            const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
            const float* positions = reinterpret_cast<const float*>(&(model.buffers[posView.buffer].data[posView.byteOffset + posAccessor.byteOffset]));

            std::cout << "First triangle vertices:" << std::endl;
            for (int v = 0; v < 9; v += 3) {
                std::cout << "v" << v / 3 << ": ("
                    << positions[v] << ", "
                    << positions[v + 1] << ", "
                    << positions[v + 2] << ")" << std::endl;
            }

            // Print bounding box
            std::cout << "Bounding box:" << std::endl;
            std::cout << "Min: ("
                << posAccessor.minValues[0] << ", "
                << posAccessor.minValues[1] << ", "
                << posAccessor.minValues[2] << ")" << std::endl;
            std::cout << "Max: ("
                << posAccessor.maxValues[0] << ", "
                << posAccessor.maxValues[1] << ", "
                << posAccessor.maxValues[2] << ")" << std::endl;
        }
    }

    std::cout << "\nMesh Statistics:" << std::endl;
    std::cout << "- Total Primitives: " << totalPrimitives << std::endl;
    std::cout << "- Total Vertices: " << totalVertices << std::endl;
    std::cout << "- Total Indices: " << totalIndices << std::endl;
    std::cout << std::endl;

    // Print Camera Information
    std::cout << "Camera Information:" << std::endl;
    if (!model.cameras.empty()) {
        for (size_t i = 0; i < model.cameras.size(); i++) {
            const auto& camera = model.cameras[i];
            std::cout << "Camera " << i << ": " << (camera.name.empty() ? "unnamed" : camera.name) << std::endl;
            std::cout << "  - Type: " << camera.type << std::endl;

            if (camera.type == "perspective") {
                std::cout << "  - FOV: " << camera.perspective.yfov * (180.0f / PI) << " degrees" << std::endl;
                std::cout << "  - Aspect Ratio: " << camera.perspective.aspectRatio << std::endl;
                std::cout << "  - Near: " << camera.perspective.znear << std::endl;
                if (camera.perspective.zfar > 0) {
                    std::cout << "  - Far: " << camera.perspective.zfar << std::endl;
                }
            }
        }
    }
    else {
        std::cout << "No cameras found in GLTF file. Using default camera." << std::endl;
        setupDefaultCamera();
    }
    std::cout << std::endl;

    std::cout << "\n=== GLTF Loading Complete ===\n" << std::endl;
}

void Scene::setupDefaultCamera() {
    Camera& cam = state.camera;

    cam.resolution = glm::vec2(800, 600);
    cam.position = glm::vec3(0, 0, -3);
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

    // Add this after camera setup
    printf("Camera setup:\n");
    printf("Position: (%f,%f,%f)\n", cam.position.x, cam.position.y, cam.position.z);
    printf("View: (%f,%f,%f)\n", cam.view.x, cam.view.y, cam.view.z);
    printf("Resolution: %dx%d\n", cam.resolution.x, cam.resolution.y);
}