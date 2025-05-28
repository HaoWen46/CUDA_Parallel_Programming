#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int MAX = 10000000;
const double eps = 1.0e-10;
const double Ke = 8.988e9;
float *h_phi[2], *h_C;
float *d_phi[2], *d_C;

__global__ void Laplacian(float* phi_old, float *phi_new, float* C) {
    extern __shared__ float cache[];

    float t, l, r, b, u, d, diff = 0.0;
    int Nx = blockDim.x * gridDim.x;
    int Ny = blockDim.y * gridDim.y;
    int Nz = blockDim.z * gridDim.z;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;
    int cacheIndex = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    int site = x + Nx * (y + Ny * z);
    if (x && y && z && x < Nx - 1 && y < Ny - 1 && z < Nz - 1) {
        d = phi_old[site - Nx * Ny];
        u = phi_old[site + Nx * Ny];
        b = phi_old[site - Nx];
        t = phi_old[site + Nx];
        l = phi_old[site - 1];
        r = phi_old[site + 1];
        phi_new[site] = (b + t + l + r + u + d) / 6.0;
        diff = phi_new[site] - phi_old[site];
    }
    
    cache[cacheIndex] = diff * diff;
    __syncthreads();

    int ib = blockDim.x * blockDim.y * blockDim.z / 2;
    while (ib != 0) {
        if (cacheIndex < ib) {
            cache[cacheIndex] += cache[cacheIndex + ib];
        }
        __syncthreads();
        ib /= 2;
    }
    int blockIndex = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (cacheIndex == 0) C[blockIndex] = cache[0];
}

int main(void) {
    int gid = 0;
    if (cudaSetDevice(gid) != cudaSuccess) {
        printf("GPU failed\n");
        exit(1);
    }

    int L;
    printf("Enter the number of L:");
    scanf("%d", &L);
    printf("%d\n", L);

    int tx, ty, tz;
    printf("Enter the number of threads (tx, ty, tz) per block: ");
    scanf("%d%d%d", &tx, &ty, &tz);
    printf("%d %d %d\n", tx, ty, tz);

    int threadsPerBlock = tx * ty * tz;
    printf("The number of threads per block: %d\n", threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024 ! \n");
        exit(0);
    }

    dim3 threads(tx, ty, tz);

    //int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    //printf("The number of blocks is %d\n", blocksPerGrid);

    int bx = L / tx;
    int by = L / ty;
    int bz = L / tz;
    if (bx * tx != L || by * ty != L || bz * tz != L) {
        printf("The block size is incorrect\n");
        exit(0);
    }

    int blocksPerGrid = bx * by * bz;
    printf("The number of blocks per grid: %d\n", blocksPerGrid);

    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(0);
    }

    dim3 blocks(bx, by, bz);
    printf("The dimension of the grid is (%d, %d, %d)\n", bx, by, bz);

    int size = L * L * L * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);

    h_phi[0] = (float*)malloc(size);
    h_phi[1] = (float*)malloc(size);
    h_C = (float*)malloc(sb);

    memset(h_phi[0], 0, size);
    memset(h_phi[1], 0, size);

    int center = L / 2 + L * (L / 2 + L * L / 2);
    h_phi[0][center] = 1.0;

    /*
    FILE *out1;
    out1 = fopen("initial.dat", "w");

    fprintf(out1, "initial field configuration:\n");
    for (int k = L - 1; k >= 0; k--) {
        for (int j = L - 1; j >= 0; j--) {
            for (int i = 0; i < L; i++) {
                fprintf(out1, "%.2e ", h_phi[0][i + j * L + k * L * L]);
            }
            fprintf(out1, "\n");
        }
        fprintf(out1, "--------------------------------\n");
    }

    fclose(out1);
    */

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&d_phi[0], size);
    cudaMalloc((void**)&d_phi[1], size);
    cudaMalloc((void**)&d_C, sb);

    cudaMemcpy(d_phi[0], h_phi[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi[1], h_phi[1], size, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float InTime;
    cudaEventElapsedTime(&InTime, start, stop);
    printf("Input time for GPU: %f (ms) \n", InTime);

    cudaEventRecord(start, 0);

    int sm = tx * ty * tz * sizeof(float);

    int iter;
    float error = eps + 1.0;
    for (iter = 0; error > eps && iter < MAX; iter++) {
        int turn = iter & 1;
        Laplacian<<<blocks, threads, sm>>>(d_phi[turn], d_phi[!turn], d_C);
        cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);

        error = 0.0;
        for (int i = 0; i < blocksPerGrid; i++) {
            error += h_C[i];
        }
        error = sqrt(error);
    }


    printf("error (GPU) = %.15e\n",error);
    printf("total iterations (GPU) = %d\n",iter);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("Processing time for GPU: %f (ms) \n", gpuTime);
    printf("GPU Gflops: %f\n", 7.0 * (L - 2) * (L - 2) * (L - 2) * iter / (1000000.0 * gpuTime));

    cudaEventRecord(start, 0);

    cudaMemcpy(h_phi[0], d_phi[iter & 1], size, cudaMemcpyDeviceToHost);

    cudaFree(d_phi[0]);
    cudaFree(d_phi[1]);
    cudaFree(d_C);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float OutTime;
    cudaEventElapsedTime(&OutTime, start, stop);
    printf("Output time for GPU: %f (ms) \n", OutTime);

    float gputime_tot;
    gputime_tot = InTime + gpuTime + OutTime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);

    /*
    FILE *outg;
    outg = fopen("GPU.dat", "w");

    fprintf(outg, "GPU field configuration:\n");
    for (int k = L - 1; k >= 0; k--) {
        for (int j = L - 1; j >= 0; j--) {
            for (int i = 0; i < L; i++) {   
                fprintf(outg, "%.2e ", h_phi[0][i + j * L + k * L * L]);
            }
            fprintf(outg, "\n");
        }
        fprintf(out1, "--------------------------------\n");
    }
    fclose(outg);
    */

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Check result:\n");
    double sum = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0 ; k < L; k++) {
                double r = pow(i - L / 2, 2) + pow(j - L / 2, 2) + pow(k - L / 2, 2);
                if (r == 0.0) continue;
                double err = h_phi[0][i + j * L + k * L * L] - 1.0 / sqrt(r);
                sum += err * err;
            }
        }
    }
    printf("error=%20.15e\n", sum / (L * L * L));

    free(h_phi[0]);
    free(h_phi[1]);
    free(h_C);

    cudaDeviceReset();
}