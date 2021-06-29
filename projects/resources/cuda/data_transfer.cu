#include <iostream>
#include <stdio.h>
#include <nvml.h>
#include <cuda_runtime.h>

#define N 1000000000
#define NGPU 8

float D2D_copy (size_t size, int from, int to)
{
  int *pointers[2];

  cudaSetDevice (from);
  cudaDeviceEnablePeerAccess (to, 0);
  cudaMalloc (&pointers[0], size);

  cudaSetDevice (to);
  cudaDeviceEnablePeerAccess (from, 0);
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

  cudaSetDevice (from);
  cudaFree (pointers[0]);

  cudaSetDevice (to);
  cudaFree (pointers[1]);

  cudaEventDestroy (end);
  cudaEventDestroy (begin);
  cudaSetDevice (from);

  return elapsed;
}

float HToD_copy (size_t size, int deviceID)
{
    int *pointer, *d_pointer;


    cudaSetDevice (deviceID);
    cudaMalloc (&d_pointer, size);
    pointer = (int*)malloc(size);

    cudaEvent_t begin, end;
    cudaEventCreate (&begin);
    cudaEventCreate (&end);

    cudaEventRecord (begin);


    cudaMemcpyAsync (d_pointer, pointer, size, cudaMemcpyHostToDevice);
    cudaEventRecord (end);
    cudaEventSynchronize (end);

    float elapsed;
    cudaEventElapsedTime (&elapsed, begin, end);
    elapsed /= 1000;

    cudaSetDevice (deviceID);
    cudaFree (d_pointer);

    cudaEventDestroy (end);
    cudaEventDestroy (begin);

    return elapsed;
}



void printDeviceAttribute(){
    int attr_val_device_0 = 0;
    int attr_val_device_1 = 0;
    
    cudaDeviceGetAttribute(&attr_val_device_0,cudaDevAttrConcurrentManagedAccess, 0);
    cudaDeviceGetAttribute(&attr_val_device_1,cudaDevAttrConcurrentManagedAccess, 1);

    printf("concurrent managed access device 0: %d \nconcurrent managed access device 1: %d\n", attr_val_device_0, attr_val_device_1);

    int can_access_peer_device_0 = 0;
    int can_access_peer_device_1 = 0;
    cudaDeviceCanAccessPeer(&can_access_peer_device_0,0,1);
    cudaDeviceCanAccessPeer(&can_access_peer_device_1,1,0);
    printf("concurrent peer access device 0->1: %d \nconcurrent peer access device 1->0: %d\n", can_access_peer_device_0, can_access_peer_device_1);


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

// void nvlinkTest(){

//     unsigned int devices_count;

//     nvmlInit ();
//     nvmlDeviceGetCount (&devices_count);

//     nvmlDevice_t device;
//     nvmlDeviceGetHandleByIndex (0, &device);

//     nvmlNvLinkUtilizationControl_t utilization_control;
//     utilization_control.units = NVML_NVLINK_COUNTER_UNIT_BYTES;
//     utilization_control.pktfilter = NVML_NVLINK_COUNTER_PKTFILTER_ALL;
//     nvmlDeviceFreezeNvLinkUtilizationCounter (device, 0, 0, NVML_FEATURE_DISABLED);
//     nvmlDeviceSetNvLinkUtilizationControl (device, 0, 0, &utilization_control, 1);

//     unsigned long long int tx_before;
//     unsigned long long int rx_before;
//     nvmlDeviceGetNvLinkUtilizationCounter (device, 0, 0, &rx_before, &tx_before);

//     // code to measure
//     float time_third = p2p_copy(1000000000);
//     printf("time spend %f \n",time_third);

//     unsigned long long int tx_after;
//     unsigned long long int rx_after;
//     nvmlDeviceGetNvLinkUtilizationCounter (device, 0, 0, &rx_after, &tx_after);

//     const unsigned long long int tx = tx_after - tx_before;
//     const unsigned long long int rx = rx_after - rx_before;
// }


void linktest(){
    unsigned int devices_count {};

    nvmlInit ();
    nvmlDeviceGetCount (&devices_count);

    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex (1, &device);

    nvmlFieldValue_t field;
    field.scopeId = 0;
    field.fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
    nvmlDeviceGetFieldValues (device, 1, &field);
    const unsigned long long int initial_tx = field.value.ullVal;

    field.fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
    nvmlDeviceGetFieldValues (device, 1, &field);
    const unsigned long long int initial_rx = field.value.ullVal;

    // code to measure
    float time_third = D2D_copy(N, 0, 1);
    printf("time spend %f \n",time_third);

    field.fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
    nvmlDeviceGetFieldValues (device, 1, &field);
    const unsigned long long int final_tx = field.value.ullVal;

    field.fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
    nvmlDeviceGetFieldValues (device, 1, &field);
    const unsigned long long int final_rx = field.value.ullVal;

    const unsigned int rx = final_rx - initial_rx;
    const unsigned int tx = final_tx - initial_tx;
    printf("rx : %d\n", rx);
    printf("tx : %d\n", tx);
}

int main(){
    size_t size = N; // 1_000_000_000
    // float time_first = p2p_copy(size);
    // printf("time spend %f \n",time_first);


    for(int i = 0; i<NGPU; i++){
        for(int j = 0 ; j<NGPU; j++){
            float time_first = D2D_copy(size, i, j);
            printf("from: %d, to: %d, time spend %f, transfer rate: %f GB/s \n",i, j, time_first, 1/time_first);
        }
    }
    // linktest();

    // printDeviceAttribute();
    // float time_third = p2p_copy(N);
    // printf("time spend %f \n",time_third);

    // // disable peer access
    // cudaError_t err;
    // cudaSetDevice(0);
    // err = cudaDeviceDisablePeerAccess(1);
    // printf("err: %s\n", cudaGetErrorString(err));

    // cudaSetDevice(1);
    // err = cudaDeviceDisablePeerAccess(0);
    // printf("err: %s\n", cudaGetErrorString(err));
    // printDeviceAttribute();

    // cudaSetDevice(0);
    // linktest();


    float time_first = HToD_copy(size, 1);
    printf("time spend HToD %f \n",time_first);
    return 0;
}
