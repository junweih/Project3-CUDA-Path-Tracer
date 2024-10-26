#include "main.h"
#include "preview.h"
#include <cstring>
#include <direct.h>
#include <iomanip>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];

	//testGLTFLoading(sceneFile);
	//return 0;

	// Load scene file
	scene = new Scene(sceneFile);

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	// Initialize CUDA and GL components
	init();

	// Initialize ImGui Data
	InitImguiData(guiData);
	InitDataContainer(guiData);

	// GLFW main loop
	mainLoop();

	return 0;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
		}
	}

	std::string filename = renderState->imageName;
	std::string filepath = "../img/" + filename + "/";
	std::ostringstream ss;
	ss << filepath << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// Create directory if it doesn't exist
	_mkdir(filepath.c_str());


	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathtraceFree();
		pathtraceInit(scene);
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		pathtrace(pbo_dptr, frame, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		saveImage();
		pathtraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			renderState = &scene->state;
			Camera& cam = renderState->camera;
			cam.lookAt = ogLookAt;
			break;
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}

// Utility function to print vector values
void printVec3(const glm::vec3& v, const std::string& label) {
	std::cout << label << ": ("
		<< std::fixed << std::setprecision(3)
		<< v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
}

// Test loading and basic validation of a GLTF scene
void testGLTFLoading(const std::string& filename) {
	std::cout << "\n=== Testing GLTF Loading: " << filename << " ===" << std::endl;

	try {
		Scene scene;
		scene.loadFromFile(filename);

		// 1. Basic Scene Validation
		std::cout << "\n1. Scene Statistics:" << std::endl;
		std::cout << "Number of geometries: " << scene.geoms.size() << std::endl;
		std::cout << "Number of materials: " << scene.materials.size() << std::endl;

		// Verify we have at least one geometry and material
		assert(!scene.geoms.empty() && "Scene should have at least one geometry");
		assert(!scene.materials.empty() && "Scene should have at least one material");

		// 2. Geometry Validation
		std::cout << "\n2. Geometry Validation:" << std::endl;
		for (size_t i = 0; i < scene.geoms.size(); i++) {
			const Geom& geom = scene.geoms[i];
			std::cout << "\nGeometry " << i << ":" << std::endl;

			// Check geometry type
			std::cout << "Type: " <<
				(geom.type == SPHERE ? "Sphere" :
					geom.type == CUBE ? "Cube" :
					geom.type == GLTF_MESH ? "GLTF Mesh" : "Unknown") << std::endl;

			// Validate material ID
			assert(geom.materialid >= 0 &&
				geom.materialid < scene.materials.size() &&
				"Material ID should be valid");
			std::cout << "Material ID: " << geom.materialid << std::endl;

			// Print transformation info
			printVec3(geom.translation, "Translation");
			printVec3(geom.rotation, "Rotation");
			printVec3(geom.scale, "Scale");

			// For mesh type, validate mesh data
			if (geom.type == GLTF_MESH) {
				assert(geom.meshData != nullptr && "Mesh data should not be null");
				assert(geom.meshData->numTriangles > 0 && "Should have triangles");
				assert(geom.meshData->dev_triangles != nullptr && "Triangle data should be allocated");

				std::cout << "Number of triangles: " << geom.meshData->numTriangles << std::endl;
			}

			// Validate transformation matrices
			assert(glm::length(glm::vec3(geom.transform[3])) >= 0 &&
				"Transform matrix should be valid");
		}

		// 3. Material Validation
		std::cout << "\n3. Material Validation:" << std::endl;
		for (size_t i = 0; i < scene.materials.size(); i++) {
			const Material& material = scene.materials[i];
			std::cout << "\nMaterial " << i << ":" << std::endl;

			// Check color values are valid
			assert(material.color.r >= 0 && material.color.r <= 1 &&
				material.color.g >= 0 && material.color.g <= 1 &&
				material.color.b >= 0 && material.color.b <= 1 &&
				"Color values should be in range [0,1]");

			printVec3(material.color, "Base Color");

			// Print material properties
			std::cout << "Specular Exponent: " << material.specular.exponent << std::endl;
			printVec3(material.specular.color, "Specular Color");
			std::cout << "Is Reflective: " << (material.hasReflective ? "Yes" : "No") << std::endl;
			std::cout << "Is Refractive: " << (material.hasRefractive ? "Yes" : "No") << std::endl;
			std::cout << "IOR: " << material.indexOfRefraction << std::endl;
			std::cout << "Emittance: " << material.emittance << std::endl;
		}

		// 4. Camera Validation
		std::cout << "\n4. Camera Validation:" << std::endl;
		const Camera& cam = scene.state.camera;

		assert(cam.resolution.x > 0 && cam.resolution.y > 0 &&
			"Camera resolution should be positive");
		std::cout << "Resolution: " << cam.resolution.x << "x" << cam.resolution.y << std::endl;

		printVec3(cam.position, "Position");
		printVec3(cam.lookAt, "Look At");
		printVec3(cam.up, "Up Vector");

		// Validate camera vectors
		assert(glm::length(cam.view) > 0.99f && glm::length(cam.view) < 1.01f &&
			"View vector should be normalized");
		assert(glm::length(cam.up) > 0.99f && glm::length(cam.up) < 1.01f &&
			"Up vector should be normalized");
		assert(glm::length(cam.right) > 0.99f && glm::length(cam.right) < 1.01f &&
			"Right vector should be normalized");

		std::cout << "\n=== GLTF Loading Tests Passed ===" << std::endl;

	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		throw;
	}
}