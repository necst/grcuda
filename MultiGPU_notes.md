1. Timer kernel: added starting event for each computational element. On sync point, CudaElapsedTime called on startEvent and stopEvent. The execution time of the kernel is stored in the attributed of the abstract class ProfilableElement which is extended by ConfigeredKernel class.
	

2. memAdviseOption:  
	-read-mostly: implemented with the class ReadMostlyMemAdviser
	-preffered-location: implemented with the class DefaultMemAdviser
	Added in the code base just like prefetch.

3.  Policy
	- data_aware: device with most data
	- stream_aware: device with less stream
	- disjoint data aware: data aware with disjoint if two computation are going to be executed on the same device
	- disjoint: basic disjoint
	- default: assign the computation on the parent one

4. Changes in CUDARuntime:
	. additions (https://docs.nvidia.com/cuda/cuda-runtime-api/index.html):
	    - cudaMemAdvise
		- cudaDeviceCanAccessPeer
		- cudaDeviceDisablePeerAccess
		- cudaDeviceEnablePeerAccess
		- cudaEventElapsedTime
	. changes: 
		- function cudaStreamCreate (line 460) calls cudaGetDevice() on the creation of the stream, the stream is created on the device that is set at the moment of the API call
		- function cudaMemPrefetchAsync: the API call has been parameterized to accomadante multiple device IDs 
		- function loadKernel (line 1273): changed to being able to load compiled kernel onto each device context.

		- function buildKernel (line 1301): after the kernels are built, are loaded onto each device context.

		- function assertCUDAInitialized (line 1564): initialize cudaContext for each device




5. tracking array position not working: classes ArrayCoherence.java and CoherenceState.java can be eliminated

6. every new logical component has its own mock version in the tests.

7. Device.java handles stream count and updates.

8. GrCUDADeviceManager.java handles devices