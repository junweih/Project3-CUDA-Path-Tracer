#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static ShadeableIntersection* dev_first_bounce_cache = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	// Guard against null scene
	if (!scene) {
		std::cerr << "Error: Attempting to initialize with null scene" << std::endl;
		return;
	}

	// Store scene pointer first
	hst_scene = scene;

	// Now we can safely access the camera
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// Create a host-side copy of geometry data to handle mesh pointers
	std::vector<Geom> geomsCopy = scene->geoms;

	// Create a temporary array to store mesh data pointers
	std::vector<MeshData> meshDataCopy;

	// Process mesh data and update pointers
	for (size_t i = 0; i < geomsCopy.size(); i++) {
		if (geomsCopy[i].type == GLTF_MESH && geomsCopy[i].meshData != nullptr) {
			// Copy the mesh data structure
			MeshData meshData;
			meshData.numTriangles = geomsCopy[i].meshData->numTriangles;

			// Allocate and copy triangle data to GPU
			cudaMalloc(&meshData.dev_triangles,
				meshData.numTriangles * sizeof(Triangle));
			cudaMemcpy(meshData.dev_triangles,
				scene->geoms[i].meshData->dev_triangles,
				meshData.numTriangles * sizeof(Triangle),
				cudaMemcpyDeviceToDevice);

			// Store the mesh data
			meshDataCopy.push_back(meshData);

			// Allocate GPU memory for the MeshData structure
			MeshData* dev_meshData;
			cudaMalloc(&dev_meshData, sizeof(MeshData));
			cudaMemcpy(dev_meshData, &meshDataCopy.back(), sizeof(MeshData),
				cudaMemcpyHostToDevice);

			// Update the geometry to point to the GPU mesh data
			geomsCopy[i].meshData = dev_meshData;
}
	}

	// Allocate and copy other buffers
	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	// Copy geometries with updated mesh pointers
	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, geomsCopy.data(),
		scene->geoms.size() * sizeof(Geom),
		cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(),
		scene->materials.size() * sizeof(Material),
		cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_first_bounce_cache, pixelcount * sizeof(ShadeableIntersection));
#endif

	cudaDeviceSynchronize();
	checkCUDAError("pathtraceInit");
}

// And update pathtraceFree to properly clean up mesh data:
void pathtraceFree() {
	cudaDeviceSynchronize();

	// Free mesh data first
	if (dev_geoms) {
		// Create a temporary array to store geometry data
		std::vector<Geom> geoms(hst_scene->geoms.size());
		cudaMemcpy(geoms.data(), dev_geoms,
			hst_scene->geoms.size() * sizeof(Geom),
			cudaMemcpyDeviceToHost);

		// Free mesh data for each geometry
		for (const auto& geom : geoms) {
			if (geom.type == GLTF_MESH && geom.meshData != nullptr) {
				// Get the mesh data from device
				MeshData meshData;
				cudaMemcpy(&meshData, geom.meshData, sizeof(MeshData),
					cudaMemcpyDeviceToHost);

				// Free triangle data
				cudaFree(meshData.dev_triangles);

				// Free the mesh data structure
				cudaFree(geom.meshData);
			}
		}

		// Free the geometries array
		cudaFree(dev_geoms);
		dev_geoms = nullptr;
	}

	// Free other buffers
	if (dev_image) {
		cudaFree(dev_image);
		dev_image = nullptr;
	}

	if (dev_paths) {
		cudaFree(dev_paths);
		dev_paths = nullptr;
	}

	if (dev_materials) {
		cudaFree(dev_materials);
		dev_materials = nullptr;
	}

	if (dev_intersections) {
		cudaFree(dev_intersections);
		dev_intersections = nullptr;
	}

#if CACHE_FIRST_BOUNCE
	if (dev_first_bounce_cache) {
		cudaFree(dev_first_bounce_cache);
		dev_first_bounce_cache = nullptr;
	}
#endif

	hst_scene = nullptr;

	cudaDeviceSynchronize();
	cudaGetLastError();
}


#pragma region Kernel
// Add this structure to store AA sample offsets
struct AAJitter {
	float x, y;
};

// generate SSAA sample offsets
__device__ void generateAAOffsets(AAJitter* offsets, int numSamples) {
	for (int i = 0; i < numSamples; ++i) {
		float angle = 2.0f * PI * i / numSamples;
		float radius = sqrtf((i + 0.5f) / numSamples);
		offsets[i].x = radius * cosf(angle) * 0.5f + 0.5f;
		offsets[i].y = radius * sinf(angle) * 0.5f + 0.5f;
	}
}


/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.color = glm::vec3(0.0f);
		segment.throughput = glm::vec3(1.0f);

		// Convert pixel coordinates to camera space
		float ndc_x = 2.0f * ((x + 0.5f) / cam.resolution.x) - 1.0f;  // Range [-1, 1]
		float ndc_y = 1.0f - 2.0f * ((y + 0.5f) / cam.resolution.y);  // Range [-1, 1]

		// Calculate view plane distances based on FOV
		float tan_fov_x = tanf(glm::radians(cam.fov.x) * 0.5f);
		float tan_fov_y = tanf(glm::radians(cam.fov.y) * 0.5f);

		// Calculate direction vector
		glm::vec3 direction = glm::normalize(
			cam.view +                    // Forward vector
			cam.right * (ndc_x * tan_fov_x) +   // Right component
			cam.up * (ndc_y * tan_fov_y)        // Up component
		);

		// Debug prints for specific pixels
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			printf("\nCamera debug info:\n");
			printf("Position: (%f,%f,%f)\n", cam.position.x, cam.position.y, cam.position.z);
			printf("View direction: (%f,%f,%f)\n", cam.view.x, cam.view.y, cam.view.z);
			printf("FOV: (%f,%f) degrees\n", cam.fov.x, cam.fov.y);
		}

		if ((x == 0 && y == 0) ||                                    // Top-left
			(x == cam.resolution.x - 1 && y == 0) ||                   // Top-right
			(x == 0 && y == cam.resolution.y - 1) ||                   // Bottom-left
			(x == cam.resolution.x - 1 && y == cam.resolution.y - 1) ||  // Bottom-right
			(x == cam.resolution.x / 2 && y == cam.resolution.y / 2))    // Center
		{
			printf("\nPixel (%d,%d):\n", x, y);
			printf("NDC coordinates: (%f,%f)\n", ndc_x, ndc_y);
			printf("Ray direction: (%f,%f,%f)\n",
				direction.x, direction.y, direction.z);
		}

		segment.ray.origin = cam.position;
		segment.ray.direction = direction;
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {
		PathSegment pathSegment = pathSegments[path_index];

		// Debug info for first thread only to avoid spam
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			printf("\n=== Starting intersection tests for path %d ===\n", path_index);
			printf("Ray origin: (%f,%f,%f)\n",
				pathSegment.ray.origin.x,
				pathSegment.ray.origin.y,
				pathSegment.ray.origin.z);
			printf("Ray direction: (%f,%f,%f)\n",
				pathSegment.ray.direction.x,
				pathSegment.ray.direction.y,
				pathSegment.ray.direction.z);
		}

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		glm::vec2 texCoord;

		for (int i = 0; i < geoms_size; i++) {
			Geom& geom = geoms[i];
			glm::vec3 tmp_intersect;
			glm::vec3 tmp_normal;
			bool tmp_outside;

			if (threadIdx.x == 0 && blockIdx.x == 0) {
				printf("\nTesting geometry %d of type %d\n", i, geom.type);
			}

			switch (geom.type) {
			case SPHERE:
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
				break;

			case CUBE:
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
				break;

			case GLTF_MESH:
				if (threadIdx.x == 0 && blockIdx.x == 0) {
					printf("Testing GLTF mesh with %d triangles\n", geom.meshData->numTriangles);
					printf("Transformation matrix:\n");
					for (int row = 0; row < 4; row++) {
						printf("%f %f %f %f\n",
							geom.transform[row][0],
							geom.transform[row][1],
							geom.transform[row][2],
							geom.transform[row][3]);
					}
				}

				glm::vec2 tmp_texcoord;
				t = meshIntersectionTest(
					geom.meshData->dev_triangles,
					geom.meshData->numTriangles,
					geom,
					pathSegment.ray,
					tmp_intersect,
					tmp_normal,
					tmp_texcoord,
					tmp_outside
				);
				break;

			default:
				t = -1.0f;
				break;
			}

			if (threadIdx.x == 0 && blockIdx.x == 0) {
				if (t > 0.0f) {
					printf("Hit found for geometry %d:\n", i);
					printf("Distance (t): %f\n", t);
					printf("Hit point: (%f,%f,%f)\n",
						tmp_intersect.x, tmp_intersect.y, tmp_intersect.z);
					printf("Normal: (%f,%f,%f)\n",
						tmp_normal.x, tmp_normal.y, tmp_normal.z);
					printf("Outside: %s\n", tmp_outside ? "true" : "false");
				}
				else {
					printf("No hit for geometry %d (t = %f)\n", i, t);
				}
			}

			if (t > 0.0f && t < t_min) {
				if (threadIdx.x == 0 && blockIdx.x == 0) {
					printf("New closest hit found! Previous t_min = %f, new t = %f\n", t_min, t);
				}
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				outside = tmp_outside;
			}
		}

		if (hit_geom_index == -1) {
			if (threadIdx.x == 0 && blockIdx.x == 0) {
				printf("\nNo intersection found for path %d\n", path_index);
			}
			intersections[path_index].t = -1.0f;
		}
		else {
			if (threadIdx.x == 0 && blockIdx.x == 0) {
				printf("\nFinal intersection result for path %d:\n", path_index);
				printf("Hit geometry: %d\n", hit_geom_index);
				printf("Material ID: %d\n", geoms[hit_geom_index].materialid);
				printf("Distance: %f\n", t_min);
				printf("Hit point: (%f,%f,%f)\n",
					intersect_point.x, intersect_point.y, intersect_point.z);
				printf("Normal: (%f,%f,%f)\n",
					normal.x, normal.y, normal.z);
				printf("=========================================\n");
			}
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				pathSegments[idx].color *= u01(rng); // apply some noise because why not
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__device__ glm::vec3 computeSkyGradient(const glm::vec3& direction) {
	// Convert direction to normalized up component
	float t = 0.5f * (direction.y + 1.0f);  // Map y from [-1,1] to [0,1]

	// Define sky colors
	glm::vec3 skyTop = glm::vec3(0.5f, 0.7f, 1.0f);     // Sky blue
	glm::vec3 skyBottom = glm::vec3(0.8f, 0.9f, 1.0f);  // Light blue/white

	// Lerp between colors based on vertical direction
	return glm::mix(skyBottom, skyTop, t);
}

/**
 * CUDA kernel for shading materials in a path tracer.
 *
 * @param iter Current iteration of the path tracer
 * @param num_paths Total number of path segments being processed
 * @param shadeableIntersections Array of intersection data for each path
 * @param pathSegments Array of path segments being traced
 * @param materials Array of material data for the scene
 * @param depth Current bounce depth in the path tracing process
 */
__global__ void shadeMaterials(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials,
	int depth
)
{
	// Calculate global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Exit if this thread is beyond the number of paths we're processing
	if (idx >= num_paths) return;

	// Exit if this path has exhausted its allowed bounces
	if (pathSegments[idx].remainingBounces == 0) return;

	// Get the intersection data for this path segment
	ShadeableIntersection intersection = shadeableIntersections[idx];

	// Process the intersection if it exists (t > 0 indicates a valid intersection)
	if (intersection.t > 0.f) {
		// Initialize random number generator for this thread
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		// Get the material properties for the intersected object
		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// Check if the intersected object is a light source
		if (material.emittance > 0.f) {
			// If it's a light, add its contribution to the path color
			pathSegments[idx].color += (materialColor * material.emittance) * pathSegments[idx].throughput;
			// Terminate the path as it has hit a light source
			pathSegments[idx].remainingBounces = 0;
		}
		else {
			pathSegments[idx].color = glm::vec3(1.0f);
			pathSegments[idx].remainingBounces = 0;

			//// For non-emissive materials, calculate the intersection point
			//glm::vec3 isect = getPointOnRay(pathSegments[idx].ray, intersection.t);
			//// Generate a new ray direction based on the material properties
			//scatterRay(pathSegments[idx], isect, intersection.surfaceNormal, material, rng);
		}
	}
	else {
		// No intersection - render sky
		glm::vec3 skyColor = computeSkyGradient(pathSegments[idx].ray.direction);
		pathSegments[idx].color = skyColor;
		pathSegments[idx].remainingBounces = 0;

	}
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct IsActivePath
{
	__host__ __device__
		bool operator()(const PathSegment& ps)
	{
		return ps.remainingBounces > 0;
	}
};

struct materialSort
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& isect1, ShadeableIntersection& isect2)
	{
		return isect1.materialId < isect2.materialId;
	}
};

#pragma endregion

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (
		cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// Tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
		if (depth == 0 && iter == 1) {
			// First iteration, first bounce: compute and cache intersections
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_first_bounce_cache);
			cudaMemcpy(dev_intersections, dev_first_bounce_cache, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else if (depth == 0) {
			// Subsequent iterations, first bounce: use cached intersections
			cudaMemcpy(dev_intersections, dev_first_bounce_cache, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			// All other bounces: compute intersections normally
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
		}
#else
		// Original code without caching
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
#endif

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

#if SORT_BY_MATERIAL
		// sort rays by material type 
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialSort());
#endif

		shadeMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			depth
			);
		checkCUDAError("shadeMaterials failed");

#if COMPACT
		// Partition paths, moving active ones to the front
		PathSegment* firstInactivePath = thrust::partition(thrust::device,
			dev_paths,
			dev_paths + num_paths,
			IsActivePath());

		// Update the count of active paths
		int numActivePaths = firstInactivePath - dev_paths;

		// Update the total number of paths we're tracking
		num_paths = numActivePaths;

		// Check termination conditions
		bool allPathsTerminated = (numActivePaths == 0);
		bool maxDepthReached = (depth >= traceDepth);

		if (allPathsTerminated || maxDepthReached) {
			iterationComplete = true;
		}
#else
		iterationComplete = (depth == traceDepth);
#endif

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
		depth++;
	}

	// remember to recover num paths
	num_paths = dev_path_end - dev_paths;
	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
