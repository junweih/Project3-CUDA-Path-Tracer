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
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
#if CACHE_FIRST_BOUNCE
	cudaMalloc(&dev_first_bounce_cache, pixelcount * sizeof(ShadeableIntersection));
#endif

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
#if CACHE_FIRST_BOUNCE
	cudaFree(dev_first_bounce_cache);
#endif

	checkCUDAError("pathtraceFree");
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

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		float u = (float)x;
		float v = (float)y;

#if SSAA
		// Stochastic Sampled Anti-Aliasing
		AAJitter aaOffsets[AA_SAMPLES];
		generateAAOffsets(aaOffsets, AA_SAMPLES);

		int sampleIndex = iter % AA_SAMPLES;

		u += aaOffsets[sampleIndex].x;
		v += aaOffsets[sampleIndex].y;

		// Add a small random jitter for additional noise reduction
		u += u01(rng) * 0.1f;
		v += u01(rng) * 0.1f;
#elif AA
		// Use a simple jittering method for anti-aliasing
		u += u01(rng);
		v += u01(rng);
#else
		// Add 0.5 to center the ray in the pixel
		u += 0.5f;
		v += 0.5f;
#endif

		glm::vec3 direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * (u - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (v - (float)cam.resolution.y * 0.5f)
		);

		if (cam.dofEnabled) {
			// Calculate the point on the focal plane
			glm::vec3 focalPoint = cam.position + cam.focalLength * direction;

			// Generate a random point on the lens for depth of field
			float r = sqrt(u01(rng)) * cam.aperture;
			float theta = u01(rng) * 2 * PI;
			glm::vec3 offset = r * (cos(theta) * cam.right + sin(theta) * cam.up);

			// Set the ray origin to the offset camera position
			segment.ray.origin = cam.position + offset;

			// Set the ray direction to point from the offset origin through the focal point
			segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);
		}
		else {
			// Standard ray generation without depth of field
			segment.ray.origin = cam.position;
			segment.ray.direction = direction;
		}


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
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
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
			// For non-emissive materials, calculate the intersection point
			glm::vec3 isect = getPointOnRay(pathSegments[idx].ray, intersection.t);
			// Generate a new ray direction based on the material properties
			scatterRay(pathSegments[idx], isect, intersection.surfaceNormal, material, rng);
		}
	}
	else {
		// If there's no intersection, set the path color to black and terminate it
		pathSegments[idx].color = glm::vec3(0.0f);
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
