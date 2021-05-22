#include <iostream>
#include <stdio.h>
#include <nvml.h>
#include <cuda_runtime.h>
float p2p_copy (size_t size)
{
  int *pointers[2];

  cudaSetDevice (0);
  cudaDeviceEnablePeerAccess (1, 0);
  cudaMalloc (&pointers[0], size);

  cudaSetDevice (1);
  cudaDeviceEnablePeerAccess (0, 0);
  cudaMalloc (&pointers[1], size);

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord (begin);
  cudaMemcpyAsync (pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
  cudaEventRecord (end);
  cudaEventSynchronize (end);

  float elapsed;
  cudaEventElapsedTime (&elapsed, begin, end);
  elapsed /= 1000;

  cudaSetDevice (0);
  cudaFree (pointers[0]);

  cudaSetDevice (1);
  cudaFree (pointers[1]);

  cudaEventDestroy (end);
  cudaEventDestroy (begin);

  return elapsed;
}

void printDeviceAttribute(){
    int attr_val_device_0 = 0;
    int attr_val_device_1 = 0;
    cudaError_t err;
    cudaDeviceGetAttribute(&attr_val_device_0,cudaDevAttrConcurrentManagedAccess, 0);
    cudaDeviceGetAttribute(&attr_val_device_1,cudaDevAttrConcurrentManagedAccess, 1);

    printf("concurrent managed access device 0: %d \nconcurrent managed access device 1: %d\n", attr_val_device_0, attr_val_device_1);

    int can_access_peer_device_0 = 0;
    int can_access_peer_device_1 = 0;
    cudaDeviceCanAccessPeer(&can_access_peer_device_0,0,1);
    cudaDeviceCanAccessPeer(&can_access_peer_device_1,1,0);
    printf("concurrent peer access device 0->1: %d \nconcurrent peer access device 1->0: %d\n", can_access_peer_device_0, can_access_peer_device_1);

    cudaSetDevice(0);
    err = cudaDeviceEnablePeerAccess(1, 0);
    printf("err: %s\n", cudaGetErrorString(err));

    cudaSetDevice(1);
    err = cudaDeviceEnablePeerAccess(0, 0);
    printf("err: %s\n", cudaGetErrorString(err));

    // cudaSetDevice(0);
    // err = cudaDeviceDisablePeerAccess(1);
    // printf("err: %s\n", cudaGetErrorString(err));

    // cudaSetDevice(1);
    // err = cudaDeviceDisablePeerAccess(0);
    // printf("err: %s\n", cudaGetErrorString(err));


    // cudaDeviceCanAccessPeer(&can_access_peer_device_0,0,1);
    // cudaDeviceCanAccessPeer(&can_access_peer_device_1,1,0);
    // printf("concurrent peer access device 0->1: %d \nconcurrent peer access device 1->0: %d\n", can_access_peer_device_0, can_access_peer_device_1);



}

void nvlinkTest(){

    // unsigned int devices_count;

    // nvmlInit ();
    // nvmlDeviceGetCount (&devices_count);

    // nvmlDevice_t device;
    // nvmlDeviceGetHandleByIndex (0, &device);

    // nvmlNvLinkUtilizationControl_t utilization_control;
    // utilization_control.units = NVML_NVLINK_COUNTER_UNIT_BYTES;
    // utilization_control.pktfilter = NVML_NVLINK_COUNTER_PKTFILTER_ALL;
    // nvmlDeviceFreezeNvLinkUtilizationCounter (device, 0, 0, NVML_FEATURE_DISABLED);
    // nvmlDeviceSetNvLinkUtilizationControl (device, 0, 0, &utilization_control, 1);

    // unsigned long long int tx_before;
    // unsigned long long int rx_before;
    // nvmlDeviceGetNvLinkUtilizationCounter (device, 0, 0, &rx_before, &tx_before);

    // // code to measure
    // float time_third = p2p_copy(1000000000);
    // printf("time spend %f \n",time_third);

    // unsigned long long int tx_after;
    // unsigned long long int rx_after;
    // nvmlDeviceGetNvLinkUtilizationCounter (device, 0, 0, &rx_after, &tx_after);

    // const unsigned long long int tx = tx_after - tx_before;
    // const unsigned long long int rx = rx_after - rx_before;
}

int main(){
    size_t size = 1000000000; // 100_000_000_000
    float time_first = p2p_copy(size);
    printf("time spend %f \n",time_first);

    float time_second = p2p_copy(size);
    printf("time spend %f \n",time_second);
    printDeviceAttribute();
    float time_third = p2p_copy(size);
    printf("time spend %f \n",time_third);
    return 0;
}