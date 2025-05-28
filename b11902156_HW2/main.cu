#include <stdio.h>
#include <stdlib.h>

const int N = 81920007;
float *h_A, *h_B, *d_A, *d_B;

void RandomInit(float*, int);

__global__ void minAbs(const float* A, float* B, const int N) {
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 1.0;
    while (i < N) {
        temp = min(temp, abs(A[i]));
        i += blockDim.x * gridDim.x;  
    }

    cache[cacheIndex] = temp;

    __syncthreads();
    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (cacheIndex < ib) {
            cache[cacheIndex] = min(cache[cacheIndex], cache[cacheIndex + ib]);
        }
        __syncthreads();
        ib /= 2;
    }
    
    if (cacheIndex == 0) {
        B[blockIdx.x] = cache[0];
    }
}

int main(void) {
    int gid = 0;
    if (cudaSetDevice(gid) != cudaSuccess) {
        printf("GPU failed\n");
        exit(1);
    }
    cudaSetDevice(gid);

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

    //int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    //printf("The number of blocks is %d\n", blocksPerGrid);

    int blocksPerGrid;
    printf("Enter the number of blocks per grid:");
    scanf("%d", &blocksPerGrid);
    printf("%d\n", blocksPerGrid);

    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(0);
    }

    int size = N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(sb);

    RandomInit(h_A, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, sb);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float InTime;
    cudaEventElapsedTime(&InTime, start, stop);
    printf("Input time for GPU: %f (ms) \n", InTime);

    cudaEventRecord(start, 0);

    int sm = threadsPerBlock * sizeof(float);
    minAbs<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A, d_B, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("Processing time for GPU: %f (ms) \n", gpuTime);
    printf("GPU Gflops: %f\n", 2 * N / (1000000.0 * gpuTime));

    cudaEventRecord(start, 0);

    cudaMemcpy(h_B, d_B, sb, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    double result1 = 100.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        result1 = min(result1, h_B[i]);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float OutTime;
    cudaEventElapsedTime(&OutTime, start, stop);
    printf("Output time for GPU: %f (ms) \n", OutTime);

    float gputime_tot;
    gputime_tot = InTime + gpuTime + OutTime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);

    cudaEventRecord(start, 0);

    double result2 = 1.0;
    for (int i = 0; i < N; i++) {
        result2 = min(result2, abs(h_A[i]));
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cpuTime;
    cudaEventElapsedTime(&cpuTime, start, stop);
    printf("Processing time for CPU: %f (ms) \n", cpuTime);
    printf("CPU Gflops: %f\n", 2 * N / (1000000.0 * cpuTime));
    printf("Speed up of GPU = %f\n\n", cpuTime / (gputime_tot));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Check result:\n");
    double diff = abs((result2 - result1)/result2 );
    printf("|(result_GPU - result_CPU)/result_CPU|=%20.15e\n",diff);
    printf("result_GPU =%20.15e\n",result1);
    printf("result_CPU =%20.15e\n",result2);
    printf("\n");

    free(h_A);
    free(h_B);

    cudaDeviceReset();
}

void RandomInit(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = 2.0 * rand() / (float)RAND_MAX - 1.0;
    }
}