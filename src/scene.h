#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <glm/glm.hpp>
#include "utilities.h"
#include "sceneStructs.h"
#include "json.hpp"  // Add this for JSON support

using json = nlohmann::json;

class Scene {
public:
    // Public member variables
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    // Constructors and destructor
    Scene();
    Scene(const std::string& filename);
    ~Scene() = default;

    // Public methods
    void loadFromFile(const std::string& filename);

private:
    // Private loading methods
    void loadJSON(const std::string& filename);
    void loadGLTF(const std::string& filename);

    // JSON loading helpers
    void loadMaterials(const json& materials);
    void loadGeoms(const json& objects);
    void loadCamera(const json& camera);

    // Utility methods
    void setupDefaultCamera();
};