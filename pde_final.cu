#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define EPS 0.0000001f
#define NTPB 1024
#define NB 16384
#define r 0.1f
#define N 1024 // maximum value

/**
 * Debug function
*/
void  print_arr(float *arr, int size)
{
  for(int i = 0; i < size; ++i)
    printf("%f ", arr[i]);
  printf("\n");
}

/**
 * Catch CUDA errors
*/
void testCUDA(cudaError_t error, const char *file, int line)  {
    if (error != cudaSuccess)
	{
       printf("There is an error in file %s at line %d\n", file, line);
       printf("%s\n", cudaGetErrorName(error));
       printf("%s\n", cudaGetErrorString(error));
       exit(EXIT_FAILURE);
    } 
}

/**
 * Question 1
 * Code to generate random values for our matrices
*/
void generateTridiagonalSystemsThomas(float* h_a_f, float* h_b_f, float* h_c_f, float* h_y_f, int systemSize, int numSystems) {
    for (int i = 0; i < numSystems; ++i) {
        for (int j = 0; j < systemSize; ++j) {
            int index = i * systemSize + j;

            // For h_a_f and h_c_f, the first element of each system is 0
            if (j == 0) {
                h_a_f[index] = 0;
                h_c_f[index] = (float)rand() / RAND_MAX;
            } 
            if (j == systemSize - 1) {
                h_a_f[index] = (float)rand() / RAND_MAX;
                h_c_f[index] = 0;
            } else {
                h_a_f[index] = (float)rand() / RAND_MAX; // Random float between 0 and 1
                h_c_f[index] = (float)rand() / RAND_MAX;
            }

            // h_b_f and h_y_f can be entirely random
            h_b_f[index] = (float)rand() / RAND_MAX;
            h_y_f[index] = (float)rand() / RAND_MAX;
        }
    }
}

void generateTridiagonalSystemsPCR(float* h_a_f, float* h_b_f, float* h_c_f, float* h_y_f, int systemSize, int numSystems) {
    for (int i = 0; i < numSystems; ++i) {
        for (int j = 0; j < systemSize; ++j) {
            int index = i * systemSize + j;

            // For h_a_f and h_c_f, the first element of each system is 0
            if (j == 0) {
                h_a_f[index] = 0;
                h_c_f[index] = 0;
            } else {
                h_a_f[index] = (float)rand() / RAND_MAX; // Random float between 0 and 1
                h_c_f[index] = (float)rand() / RAND_MAX;
            }

            // h_b_f and h_y_f can be entirely random
            h_b_f[index] = (float)rand() / RAND_MAX;
            h_y_f[index] = (float)rand() / RAND_MAX;
        }
    }
}

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

/**
 * Thomas algorithm separated on blocks
*/
__global__ void thomasAlgorithmPerBlock(float* a, float* b, float* c, float* y, float* x, int n) {
    int systemIndex = blockIdx.x * n;

    // Forward phase
    c[systemIndex] = c[systemIndex] / b[systemIndex];
    y[systemIndex] = y[systemIndex] / b[systemIndex];

    for (int i = 1; i < n; i++) {
        int idx = systemIndex + i;
        float m = 1.0 / (b[idx] - a[idx] * c[idx - 1]);
        c[idx] = c[idx] * m;
        y[idx] = (y[idx] - a[idx] * y[idx - 1]) * m;
    }

    // Backward phase
    x[systemIndex + n - 1] = y[systemIndex + n - 1];
    for (int i = n - 2; i >= 0; i--) {
        int idx = systemIndex + i;
        x[idx] = y[idx] - c[idx] * x[idx + 1];
    }
}

/**
 * Question 1
 * Thomas algorithm separated on threads
*/
__global__ void thomasAlgorithmPerThread(float* a, float* b, float* c, float* y, float* x, int n, int numSystems) {
	// We choose on which matrix we want to act
    int systemIndex = threadIdx.x * n;

    // Check if the thread is within the range of systems
    if (threadIdx.x < numSystems) {
        // Forward phase for system handled by this thread
        c[systemIndex] = c[systemIndex] / b[systemIndex];
        y[systemIndex] = y[systemIndex] / b[systemIndex];

		// We calculate the index for each step, to work on the right
		// matrix we use systemIndex
        for (int i = 1; i < n; i++) {
            int idx = systemIndex + i;
            float m = 1.0 / (b[idx] - a[idx] * c[idx - 1]);
            c[idx] = c[idx] * m;
            y[idx] = (y[idx] - a[idx] * y[idx - 1]) * m;
        }

        // Backward phase for system handled by this thread
        x[systemIndex + n - 1] = y[systemIndex + n - 1];
        for (int i = n - 2; i >= 0; i--) {
            int idx = systemIndex + i;
            x[idx] = y[idx] - c[idx] * x[idx + 1];
        }
    }
}


/**
 * Question 1
 * PCR
| b0 c0  0   0 |   | x0 |   | y0 |
| a1 b1 c1  0 | * | x1 | = | y1 |
| 0  a2 b2 c2 |   | x2 |   | y2 |
| 0   0 a3 b3 |   | x3 |   | y3 |

We have a = sa, b = sd, c = sc, y = sy
PCR method: eliminate the off-diagonal elements and reduce the system's size by half in each step
*/
__device__ void PCR_d(float* sa, float* sd, float* sc, float* sy, int* sl, int n) {

	int i, lL, d, tL, tR;
	float aL, dL, cL, yL;
	float aLp, dLp, cLp, yLp;

	// d used to determine the position in the reduced system after each iteration
	d = (n / 2 + (n % 2)) * (threadIdx.x % 2) + (int)threadIdx.x / 2;

	tL = threadIdx.x - 1;
	if (tL < 0) tL = 0;
	tR = threadIdx.x + 1;
	if (tR >= n) tR = 0;

	for (i = 0; i < (int)(logf((float)n) / logf(2.0f)) + 1; i++) {
		lL = (int)sl[threadIdx.x];
		// each thread works on reducing it's corresponding element
		aL = sa[threadIdx.x];
		dL = sd[threadIdx.x];
		cL = sc[threadIdx.x];
		yL = sy[threadIdx.x];

		dLp = sd[tL];
		cLp = sc[tL];

		// condition to ensure we don't do division by zero
		if (fabsf(aL) > EPS) {
			aLp = sa[tL];
			yLp = sy[tL];
			dL -= aL * cL / dLp;
			yL -= aL * yLp / dLp;
			aL = -aL * aLp / dLp;
			cL = -cLp * cL / dLp;
		}

		cLp = sc[tR];
		if (fabsf(cLp) > EPS) {
			aLp = sa[tR];
			dLp = sd[tR];
			yLp = sy[tR];
			dL -= cLp * aLp / dLp;
			yL -= cLp * yLp / dLp;
		}
		__syncthreads();

		if (i < (int)(logf((float)n) / logf(2.0f))) {
			sa[d] = aL;
			sd[d] = dL;
			sc[d] = cL;
			sy[d] = yL;
			sl[d] = (int)lL;
			__syncthreads();
		}
	}
	// each thread writes to the shared memory solution
	sy[(int)sl[threadIdx.x]] = yL / dL;
}

/**
 * Question 1
 * PCR wrapper so we can call PCR, since __device__ functions
 * are functions that can be called from other device or global
 * functions and are executed on the device.
 * They can't be called directly from "host".
*/
__global__ void PCR_kernel(float* sa, float* sd, float* sc, float* sy, int* sl, int n) {
    __shared__ float shared_sa[N];
    __shared__ float shared_sd[N];
    __shared__ float shared_sc[N];
    __shared__ float shared_sy[N];
    __shared__ int shared_sl[N];

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    shared_sa[threadIdx.x] = sa[idx];
    shared_sd[threadIdx.x] = sd[idx];
    shared_sc[threadIdx.x] = sc[idx];
    shared_sy[threadIdx.x] = sy[idx];
	// specify the thread we are working on since sl is based on indexes
    shared_sl[threadIdx.x] = threadIdx.x;
    __syncthreads();

	// check if we are within the correct range
	if (threadIdx.x < n)
        PCR_d(shared_sa, shared_sd, shared_sc, shared_sy, shared_sl, n);
    sy[idx] = shared_sy[threadIdx.x];
}

/**
 * Representation of the matrix :
 * 	| b1 c1  0  0 |
	| a2 b2 c2  0 |
    |  0 a3 b3 c3 |
    |  0  0 a4 b4 |
*/
int main()
{
    int systemSize = 4;
    int numSystems = 1024; // default value
    int totalSize = systemSize * numSystems;
    float *h_x, *h_a, *h_b, *h_c, *h_y;
    float *h_x_f, *h_a_f, *h_b_f, *h_c_f, *h_y_f, *h_l;
    float *d_a, *d_b, *d_c, *d_y, *d_x;
    float *d_a_f, *d_b_f, *d_c_f, *d_y_f;
    int *d_l;
    cudaEvent_t start, stop; // measure the time
    float ms1 = 0, ms2 = 0;
    FILE *file = fopen("timing_data2.txt", "a"); // file t
    srand(time(NULL)); // seed the random number generator

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // with a step of 20 and a fixed size of 4
    for (int i = 20; i < 1024; i+=20)
    {
      //numSystems = i;
      systemSize = i;
      totalSize = systemSize * numSystems;

      h_x = (float*)malloc(totalSize * sizeof(float));
      h_a = (float*)malloc(totalSize * sizeof(float));
      h_b = (float*)malloc(totalSize * sizeof(float));
      h_c = (float*)malloc(totalSize * sizeof(float));
      h_y = (float*)malloc(totalSize * sizeof(float));
      h_l = (float*)malloc(totalSize * sizeof(float));

      generateTridiagonalSystemsThomas(h_a, h_b, h_c, h_y, systemSize, numSystems);

      h_a_f = (float*)malloc(totalSize * sizeof(float));
      h_b_f = (float*)malloc(totalSize * sizeof(float));
      h_c_f = (float*)malloc(totalSize * sizeof(float));
      h_y_f = (float*)malloc(totalSize * sizeof(float));

      generateTridiagonalSystemsPCR(h_a_f, h_b_f, h_c_f, h_y_f, systemSize, numSystems);

      // h_l_f tracks the indices of the system
	    // it will reflect the new positions of the reduced elements
	    // we just initialize it to {0,1,2,3}
      //for (int j = 0; j < totalSize; j++)
    	//  h_l[j] = j;

      // Thomas'
      testCUDA(cudaMalloc((void**)&d_a, totalSize * sizeof(float)));
      testCUDA(cudaMalloc((void**)&d_b, totalSize * sizeof(float)));
      testCUDA(cudaMalloc((void**)&d_c, totalSize * sizeof(float)));
      testCUDA(cudaMalloc((void**)&d_y, totalSize * sizeof(float)));
      testCUDA(cudaMalloc((void**)&d_x, totalSize * sizeof(float)));

      testCUDA(cudaMemcpy(d_a, h_a, totalSize * sizeof(float), cudaMemcpyHostToDevice));
      testCUDA(cudaMemcpy(d_b, h_b, totalSize * sizeof(float), cudaMemcpyHostToDevice));
      testCUDA(cudaMemcpy(d_c, h_c, totalSize * sizeof(float), cudaMemcpyHostToDevice));
      testCUDA(cudaMemcpy(d_y, h_y, totalSize * sizeof(float), cudaMemcpyHostToDevice));
      testCUDA(cudaMemcpy(d_x, h_l, totalSize * sizeof(float), cudaMemcpyHostToDevice));

      testCUDA(cudaEventRecord(start, 0));
      thomasAlgorithmPerThread<<<1, numSystems>>>(d_a, d_b, d_c, d_y, d_x, systemSize, numSystems);
      testCUDA(cudaEventRecord(stop, 0));
      cudaEventSynchronize(stop);
      testCUDA(cudaDeviceSynchronize()); // wait for the kernel to complete

      testCUDA(cudaEventElapsedTime(&ms1, start, stop));

      testCUDA(cudaMemcpy(h_x, d_x, totalSize * sizeof(float), cudaMemcpyDeviceToHost));

      // PCR
      testCUDA(cudaMalloc((void**)&d_a_f, totalSize * sizeof(float)));
      testCUDA(cudaMalloc((void**)&d_b_f, totalSize * sizeof(float)));
      testCUDA(cudaMalloc((void**)&d_c_f, totalSize * sizeof(float)));
      testCUDA(cudaMalloc((void**)&d_y_f, totalSize * sizeof(float)));
      testCUDA(cudaMalloc((void**)&d_l, totalSize * sizeof(int)));

      testCUDA(cudaMemcpy(d_a_f, h_a_f, totalSize * sizeof(float), cudaMemcpyHostToDevice));
      testCUDA(cudaMemcpy(d_b_f, h_b_f, totalSize * sizeof(float), cudaMemcpyHostToDevice));
      testCUDA(cudaMemcpy(d_c_f, h_c_f, totalSize * sizeof(float), cudaMemcpyHostToDevice));
      testCUDA(cudaMemcpy(d_y_f, h_y_f, totalSize * sizeof(float), cudaMemcpyHostToDevice));

      testCUDA(cudaEventRecord(start, 0));
      PCR_kernel<<<numSystems, systemSize>>>(d_a_f, d_b_f, d_c_f, d_y_f, d_l, systemSize);
      testCUDA(cudaEventRecord(stop, 0));
      testCUDA(cudaEventSynchronize(stop));
      testCUDA(cudaDeviceSynchronize());

      testCUDA(cudaEventElapsedTime(&ms2, start, stop));

      testCUDA(cudaMemcpy(h_y_f, d_y_f, totalSize * sizeof(float), cudaMemcpyDeviceToHost));

      printf("ms : %f %f\n", ms1, ms2);
      if (file != NULL)
        fclose(file);
      file = fopen("timing_data2.txt", "a");
      printf("ms : %f %f\n", ms1, ms2);
      if (file != NULL)
        fprintf(file, "%d %f %f\n", i, ms1, ms2);
      else
        perror("Error opening file");

      // Free allocated memory
      testCUDA(cudaFree(d_a));
      testCUDA(cudaFree(d_b));
      testCUDA(cudaFree(d_c));
      testCUDA(cudaFree(d_y));
      testCUDA(cudaFree(d_x));

	    testCUDA(cudaFree(d_a_f));
      testCUDA(cudaFree(d_b_f));
      testCUDA(cudaFree(d_c_f));
      testCUDA(cudaFree(d_y_f));
      testCUDA(cudaFree(d_l));

      free(h_a_f);
      free(h_b_f);
      free(h_c_f);
      free(h_y_f);
      free(h_l);
      free(h_a);
      free(h_b);
      free(h_c);
      free(h_y);
      free(h_x);
    }

    // Display the results
    printf("Result Thomas: ");
    for (int i = 0; i < totalSize; i++) {
        printf("%f " , h_x[i]);
    }

	printf("\nResult PCR: ");
    for (int i = 0; i < totalSize; i++) {
        printf("%f " , h_y_f[i]);
    }

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // close files
    fclose(file);
    return (0);
}