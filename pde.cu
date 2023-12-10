#include <stdio.h>
#include <math.h>

#define EPS 0.0000001f
#define NTPB 1024
#define NB 16384
#define r 0.1f
#define N 4

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
 * Code to generate random values for our matrices
*/
void fillArrayWithRandomValues(float* array, int size) {
    for (int i = 0; i < size; i++)
        array[i] = (float)rand() / RAND_MAX * 10.0; // Random float between 0 and 10
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
 * PCR wrapper so we can call PCR, since __device__ functions
 * are functions that can be called from other device or global
 * functions and are executed on the device.
 * They can't be called directly from "host".
*/

#define N 4

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
    const int n = 8;
    float h_x[n];
	int h_l[n];

	// 2 matrices in one (simple matrices 4 * 4)
    float h_a[n] = {0, 1, 1, 1, 0, 1, 1, 1};
    float h_b[n] = {4, 4, 4, 4, 4, 4, 4, 4};
    float h_c[n] = {1, 2, 3, 0, 1, 2, 3, 0};
    float h_y[n] = {1, 2, 2, 1, 1, 2, 2, 1};

	srand(time(NULL)); // Seed the random number generator

	// values for Thomas' algorithm
  	// in our implementation, a must be of the form {0, _, _, _} and c
  	// must be {0, _, _, _}
  	// 
	// float h_a[n] = {0, 1, 1, 1};
  	// float h_b[n] = {4, 4, 4, 4};
  	// float h_c[n] = {0, 1, 2, 3};
  	// float h_y[n] = {1, 2, 2, 1};
	// fill with random values if you want to try different results
    // fillArrayWithRandomValues(h_a, n);
    // fillArrayWithRandomValues(h_b, n);
    // fillArrayWithRandomValues(h_c, n);
    // fillArrayWithRandomValues(h_y, n);

	// values for PCR algorithm
	float h_a_f[] = {0, 1, 1, 1, 0, 1, 1, 1};
	float h_b_f[] = {4, 4, 4, 4, 4, 4, 4, 4};
	float h_c_f[] = {0, 1, 2, 3, 0, 1, 2, 3};
	float h_y_f[] = {1, 2, 2, 1, 1, 2, 2, 1};
	
  	// float h_a_f[] = {0, 1, 1, 1};
	// float h_b_f[] = {4, 4, 4, 4};
	// float h_c_f[] = {0, 1, 2, 3};
	// float h_y_f[] = {1, 2, 2, 1};
	// h_l_f tracks the indices of the system
	// it will reflect the new positions of the reduced elements
	// we just initialize it to {0,1,2,3}
	for (int i = 0; i < n; i++)
    	h_l[i] = i;

    float *d_a, *d_b, *d_c, *d_y, *d_x;
	
	// Thomas'
    testCUDA(cudaMalloc((void**)&d_a, n * sizeof(float)));
    testCUDA(cudaMalloc((void**)&d_b, n * sizeof(float)));
    testCUDA(cudaMalloc((void**)&d_c, n * sizeof(float)));
    testCUDA(cudaMalloc((void**)&d_y, n * sizeof(float)));
    testCUDA(cudaMalloc((void**)&d_x, n * sizeof(float)));

	testCUDA(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_c, h_c, n * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_x, h_l, n * sizeof(float), cudaMemcpyHostToDevice));

	int numSystems = 2;
  const int systemSize = n / numSystems;
    thomasAlgorithmPerThread<<<1, numSystems>>>(d_a, d_b, d_c, d_y, d_x, systemSize, numSystems);
    cudaDeviceSynchronize();
	testCUDA(cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

	// PCR
	float *d_a_f, *d_b_f, *d_c_f, *d_y_f;
	int *d_l;
	testCUDA(cudaMalloc((void**)&d_a_f, n * sizeof(float)));
	testCUDA(cudaMalloc((void**)&d_b_f, n * sizeof(float)));
	testCUDA(cudaMalloc((void**)&d_c_f, n * sizeof(float)));
	testCUDA(cudaMalloc((void**)&d_y_f, n * sizeof(float)));
	testCUDA(cudaMalloc((void**)&d_l, n * sizeof(int)));

    testCUDA(cudaMemcpy(d_a_f, h_a_f, n * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_b_f, h_b_f, n * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_c_f, h_c_f, n * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_y_f, h_y_f, n * sizeof(float), cudaMemcpyHostToDevice));

 	PCR_kernel<<<numSystems, systemSize>>>(d_a_f, d_b_f, d_c_f, d_y_f, d_l, systemSize);
	cudaDeviceSynchronize();
	testCUDA(cudaMemcpy(h_y_f, d_y_f, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Display the results
    printf("Result Thomas: ");
    for (int i = 0; i < n; i++) {
        printf("%f " , h_x[i]);
    }

	printf("\nResult PCR: ");
    for (int i = 0; i < n; i++) {
        printf("%f " , h_y_f[i]);
    }

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

    return (0);
}