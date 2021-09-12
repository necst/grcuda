# Summary of changes to support multi GPU scheduling

1. Timer for kernel: added starting event for each computational element. On sync point, CudaElapsedTime called on startEvent and stopEvent. The execution time of the kernel is stored in the attribute of the abstract class ProfilableElement which is extended by ConfiguredKernel class. Times are stored in a map that associates each device to its execution times. Currently, only the last execution time for each device is stored.
	

2. `memAdviseOption`:  
	* `read-mostly`: implemented with the class ReadMostlyMemAdviser
	* `preffered-location` (sic): implemented with the class DefaultMemAdviser. It behaves similarly to prefetch: the MemAdvise is applied on each array argument, for every kernel. This is definitely not necessary, we should track what advices are currently active for a given array-device pair, and avoid repeating them (as it likely has a cost in terms of CUDA API calls).

3.  Policies for multi-GPU
   	* `disjoint`: the basic disjoint policy (with no support for multiple GPUs). If a parent has more children, each children has a different stream.
	* `default`: assign the computation on the parent one (and execute it on the same device as the parent)
	* `data_aware`: assign the computation to the device with most data
	* `stream_aware`: assign the computation to device with fewer streams
	* `disjoint_data_aware`: data-aware with disjoint if two computation are executed on the same device

4. Changes in CUDARuntime:
	* New APIs addied to the runtime (https://docs.nvidia.com/cuda/cuda-runtime-api/index.html):
		1. cudaMemAdvise
		2. cudaDeviceCanAccessPeer
		3. cudaDeviceDisablePeerAccess
		4. cudaDeviceEnablePeerAccess
		5. cudaEventElapsedTime
	* Changes to existing APIs: 
		1. Function cudaStreamCreate (line 460) calls cudaGetDevice() on the creation of the stream. The stream is created on the device that is set at the moment of the API call
		2. Function cudaMemPrefetchAsync: the API now allows specifying the device ID
		3. Function loadKernel (line 1273): changed to load compiled kernel onto each device context.
		4. Function buildKernel (line 1301): after the kernels are built, kernels are loaded onto each device context.
		5. Function assertCUDAInitialized (line 1564): initialize cudaContext for each device

5. Tracking array position is not working: classes ArrayCoherence.java and CoherenceState.java can be eliminated. **This means that we currently don't know which GPU has updated data**

6. Every new logical component has its own mock version in the tests.

7. Device.java tracks which streams are available to it, and which are free.

8. GrCUDADeviceManager.java handles devices, and provides APIs to query and update their state.
