#ifndef JSON_SCENE_LOADER_H
#define JSON_SCENE_LOADER_H

#include <string>
#include <json.hpp>
#include "scene.h"

using json = nlohmann::json;

class JSONSceneLoader {
public:
    static Scene* loadScene(const std::string& filename);

private:
    static void loadMaterials(const json& materials, Scene* scene);
    static void loadJsonObjects(const json& objects, Scene* scene);
    static void loadCamera(const json& camera, Scene* scene);
    static void loadGLTF(Scene* scene, const std::string& filename);
    
};

#endif // JSON_SCENE_LOADER_H