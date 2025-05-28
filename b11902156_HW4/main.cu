#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

const int N = 40960000;
float *h_A, *h_B, *h_C;

void RandomInit(float*, int);

__global__ void VecDot(const float* A, const float* B, float* C, const int N) {
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0;
    while (i < N) {
        temp += A[i] * B[i];
        i += blockDim.x * gridDim.x;  
    }

    cache[cacheIndex] = temp;

    __syncthreads();
    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (cacheIndex < ib) {
            cache[cacheIndex] += cache[cacheIndex + ib];
        }
        __syncthreads();
        ib /= 2;
    }
    
    if (cacheIndex == 0) {
        C[blockIdx.x] = cache[0];
    }
}

int main(void) {
    int NGPU; 
    printf("Enter the number of GPUs:");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);
    if (NGPU <= 0) {
        printf("number of GPUs must be greater than 0\n");
        exit(0);
    }
    int *dev = (int*)malloc(NGPU * sizeof(int));
    for (int i = 0; i < NGPU; i++) {
        dev[i] = i;
    }

    int cpu_thread_id = 0;

    int threadsPerBlock;
    printf("Enter the number of threads per block:");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024 ! \n");
        exit(0);
    }
    if (!threadsPerBlock || (threadsPerBlock & (threadsPerBlock - 1))) {
        printf("The number of threads per block must be a power of 2!\n");
        exit(0);
    }

    int blocksPerGrid;
    printf("Enter the number of blocks per grid:");
    scanf("%d", &blocksPerGrid);
    printf("%d\n", blocksPerGrid);

    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(0);
    }

    int size = N * sizeof(float);
    int sb = blocksPerGrid * NGPU * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(sb);

    if (!h_A || !h_B || !h_C) {
        printf("Not enough of memory\n");
        exit(1);
    }

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    cudaEvent_t start, stop;

    float InTime, gpuTime, OutTime;

    double result1[2] = {0.0, 0.0};
    double result2 = 0.0;

    omp_set_num_threads(NGPU);

#pragma omp parallel private(cpu_thread_id) 
{
    float *d_A, *d_B, *d_C;

    cpu_thread_id = omp_get_thread_num();
    cudaSetDevice(dev[cpu_thread_id]);
    
    if (cpu_thread_id == 0) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    cudaMalloc((void**)&d_A, size / NGPU);
    cudaMalloc((void**)&d_B, size / NGPU);
    cudaMalloc((void**)&d_C, sb / NGPU);

    cudaMemcpy(d_A, h_A + N / NGPU * cpu_thread_id, size / NGPU, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B + N / NGPU * cpu_thread_id, size / NGPU, cudaMemcpyHostToDevice);
#pragma omp barrier

    if (cpu_thread_id == 0) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&InTime, start, stop);
        printf("Input time for GPU: %f (ms) \n", InTime);

        cudaEventRecord(start, 0);        
    }

    int sm = threadsPerBlock * sizeof(float);
    VecDot<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A, d_B, d_C, N / NGPU);
    cudaDeviceSynchronize();

    if (cpu_thread_id == 0) {    
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("Processing time for GPU: %f (ms) \n", gpuTime);
        printf("GPU Gflops: %f\n", 3.0 * N / (1000000.0 * gpuTime));
        
        cudaEventRecord(start, 0);
    }

    cudaMemcpy(h_C + blocksPerGrid * cpu_thread_id, d_C, sb / NGPU, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < blocksPerGrid; i++) {
        result1[cpu_thread_id] += h_C[i + blocksPerGrid * cpu_thread_id];
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (cpu_thread_id == 0) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&OutTime, start, stop);
        printf("Output time for GPU: %f (ms) \n", OutTime);
    }
}

    float gputime_tot;
    gputime_tot = InTime + gpuTime + OutTime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);

    cudaEventRecord(start, 0);

    for (int i = 0; i < N; i++) {
        result2 += h_A[i] * h_B[i];
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cpuTime;
    cudaEventElapsedTime(&cpuTime, start, stop);
    printf("Processing time for CPU: %f (ms) \n", cpuTime);
    printf("CPU Gflops: %f\n", 3.0 * N / (1000000.0 * cpuTime));
    printf("Speed up of GPU = %f\n\n", cpuTime / (gputime_tot));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 1; i < NGPU; i++) {
        result1[0] += result1[i];
    }

    printf("Check result:\n");
    double diff = abs((result2 - result1[0]) / result2);
    printf("|(result_GPU - result_CPU)/result_CPU|=%20.15e\n", diff);
    printf("result_GPU =%20.15e\n", result1[0]);
    printf("result_CPU =%20.15e\n", result2);
    printf("\n");

    free(h_A);
    free(h_B);
    free(h_C);
    free(dev);

    cudaDeviceReset();
}

void RandomInit(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = rand() / (float)RAND_MAX;
    }
}