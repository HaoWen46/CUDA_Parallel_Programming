#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

long long N = 81920000;
float Rmin = 0;
float Rmax = 24;
float* data_h;          // host vectors
float* data_d;          // device vectors
unsigned int* hist_h;   
unsigned int* hist_d;
unsigned int* hist_c;   // CPU solution

void RandomExp(float*, int);

__global__ void hist_gmem(float* data, long N, unsigned int* hist, int bins, float Rmin, float binsize) {
    // use global memory and atomic addition
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long stride = blockDim.x * gridDim.x;

    while (i < N) {
        int index = (int)((data[i]-Rmin)/binsize);
        if (index < bins) {
            atomicAdd(&hist[index],1);
        }
        i += stride;       // goto the next grid
    }

    __syncthreads();
}

int main() {
    srand(12345);
    int gid = 0, threadsPerBlock, blocksPerGrid, sm, bins, index, bsize;
    float Intime, gputime, Outime, gputime_tot, binsize;
    long long size = N * sizeof(float);
    if (cudaSetDevice(gid) != cudaSuccess) {
        printf("GPU failed\n");
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    data_h = (float*)malloc(size);
    if(data_h == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**)&data_d, size);

    printf("Starting to generate data by RNG\n");
    fflush(stdout);

    RandomExp(data_h, N);

    printf("Finish the generaton of data\n");
    fflush(stdout);

    cudaMemcpy(data_d, data_h, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

/**********************************************************************************************/
    // gmem Part
    printf("To find the histogram of a data set (with real numbers) (GPU with gmem): \n");

    // threadsPerBlock
    printf("Enter the number of threads per block: ");
    scanf("%d",&threadsPerBlock);
    printf("%d\n",threadsPerBlock);
    fflush(stdout);
    if( threadsPerBlock > 1024 ) {
        printf("The number of threads per block must be less than 1024 ! \n");
        exit(0);
    }

    // blocksPerGrid
    printf("Enter the number of blocks per grid: ");
    scanf("%d",&blocksPerGrid);
    printf("%d\n",blocksPerGrid);
    if( blocksPerGrid > 2147483647 ) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(0);
    }
    printf("The number of blocks is %d\n", blocksPerGrid);
    fflush(stdout);

    printf("Enter the number of bins of the histogram: ");
    //scanf("%d",&bins);
    bins = 64;
    printf("%d\n",bins);
    fflush(stdout);

    bsize = bins*sizeof(int);
    binsize = (Rmax - Rmin)/(float)bins;

    hist_h = (unsigned int*)malloc(bsize);
    if(hist_h == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**)&hist_d, bsize);

    for(int i=0; i<bins; i++) {
        hist_h[i]=0;
    }

    cudaEventRecord(start, 0);

    cudaMemcpy(hist_d, hist_h, bsize, cudaMemcpyHostToDevice);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // Intime
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    cudaEventRecord(start,0);

    sm = threadsPerBlock*sizeof(int);

    hist_gmem <<< blocksPerGrid, threadsPerBlock, sm >>> (data_d, N, hist_d, bins, Rmin, binsize);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",2*N/(1000000.0*gputime));

    cudaEventRecord(start,0);

    cudaMemcpy(hist_h, hist_d, bsize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // Outime
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    // gputime_tot
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n\n",gputime_tot);

    FILE *out_gmem;            // save histogram in file
    out_gmem = fopen("hist_gmem.dat","w");

    fprintf(out_gmem, "Histogram (GPU):\n");
    for(int i=0; i<bins; i++) {
        float x=Rmin+(i+0.5)*binsize;         // the center of each bin
        fprintf(out_gmem,"%f %d \n",x,hist_h[i]);
    }
    fclose(out_gmem);

    free(hist_h);
    cudaFree(hist_d);
/**********************************************************************************************/

/**********************************************************************************************/
    // CPU part
    hist_c = (unsigned int*)malloc(bsize);
    for(int i=0; i<bins; i++) {
        hist_c[i]=0;
    }

    cudaEventRecord(start,0);

    for(int i=0; i<N; i++) {
        index = (int)((data_h[i]-Rmin)/binsize);
        if( (index > bins-1) || (index < 0)) {
            printf("data[%d]=%f, index=%d\n",i,data_h[i],index);
            exit(0);
        } 
        hist_c[index]++;
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",2*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int sum=0;
    for(int i=0; i<bins; i++) {
        sum += hist_c[i];
    }
    if(sum != N) {
        printf("Error, sum = %d\n",sum);
        exit(0);
    }

    FILE *out_cpu;            // save histogram in file
    out_cpu = fopen("hist_cpu.dat","w");

    fprintf(out_cpu, "Histogram (CPU):\n");
    for(int i=0; i<bins; i++) {
        float x=Rmin+(i+0.5)*binsize;         // the center of each bin
        fprintf(out_cpu,"%f %d \n",x,hist_c[i]);
    }
    fclose(out_cpu);
/**********************************************************************************************/

    cudaFree(data_d);
    free(data_h);
    free(hist_c);

    cudaDeviceReset();
}

void RandomExp(float* data, int n) {
    for (int i = 0; i < n; i++) {
        double y = (double)rand() / RAND_MAX;
        data[i] = -log(1.0 - y);
    }
}