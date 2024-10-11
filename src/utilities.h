#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define INV_PI            0.31830988618379067f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.0001f
#define OFFSET            0.01f

#define COMPACT 1
#define SORT_BY_MATERIAL 0

#define AA 1
#define SSAA 0
#define AA_SAMPLES 16  // Number of samples per pixel for SSAA

// Logic to control CACHE_FIRST_BOUNCE based on SSAA and AA
#if SSAA || AA
#define CACHE_FIRST_BOUNCE 0
#else
    // When both SSAA and AA are 0, keep the manual setting
#define CACHE_FIRST_BOUNCE 0  // You can manually change this to 1 when needed
#endif


class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
