#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define EPS 0.0000001f
#define NTPB 1024
#define NB 16384
#define r 0.1f
// #define N 1024 // maximum value
typedef float MyTab[NB][NTPB];

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line){

	if (error != cudaSuccess) {
	  printf("There is an error in file %s at line %d\n", file, line);
    exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

/*************************************************************************/
/*                   Black-Sholes Formula                                */
/*************************************************************************/
/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x){
  const double p= 0.2316419;
  const double b1= 0.319381530;
  const double b2= -0.356563782;
  const double b3= 1.781477937;
  const double b4= -1.821255978;
  const double b5= 1.330274429;
  const double one_over_twopi= 0.39894228;  
  double t;

  if(x >= 0.0){
	t = 1.0 / ( 1.0 + p * x );
    return (1.0 - one_over_twopi * exp( -x * x / 2.0 ) * t * ( t *( t * ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }else{/* x < 0 */
    t = 1.0 / ( 1.0 - p * x );
    return ( one_over_twopi * exp( -x * x / 2.0 ) * t * ( t *( t * ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
  }
}

/**
 * Debug function
*/
void  print_arr(float *arr, int size)
{
  for(int i = 0; i < size; ++i)
    printf("%f ", arr[i]);
  printf("\n");
}

// /**
//  * Catch CUDA errors
// */
// void testCUDA(cudaError_t error, const char *file, int line)  {
//     if (error != cudaSuccess)
// 	{
//        printf("There is an error in file %s at line %d\n", file, line);
//        printf("%s\n", cudaGetErrorName(error));
//        printf("%s\n", cudaGetErrorString(error));
//        exit(EXIT_FAILURE);
//     } 
// }

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

/**
 * Thomas algorithm separated on blocks
*/
__global__ void thomasAlgorithmPerBlock(float* a, float* b, float* c, float* y, float* x, int n)
{
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
// __global__ void PCR_kernel(float* sa, float* sd, float* sc, float* sy, int* sl, int n) {
//     __shared__ float shared_sa[N];
//     __shared__ float shared_sd[N];
//     __shared__ float shared_sc[N];
//     __shared__ float shared_sy[N];
//     __shared__ int shared_sl[N];

// 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     shared_sa[threadIdx.x] = sa[idx];
//     shared_sd[threadIdx.x] = sd[idx];
//     shared_sc[threadIdx.x] = sc[idx];
//     shared_sy[threadIdx.x] = sy[idx];
// 	// specify the thread we are working on since sl is based on indexes
//     shared_sl[threadIdx.x] = threadIdx.x;
//     __syncthreads();

// 	// check if we are within the correct range
// 	if (threadIdx.x < n)
//         PCR_d(shared_sa, shared_sd, shared_sc, shared_sy, shared_sl, n);
// 	sy[idx] = shared_sy[threadIdx.x];
// }


/**
 * Question 2
 * PDE
*/
__global__ void PDE_diff_k4(float dt, float dx, float dsig, float pmin, float pmax, float sigmin, int N, MyTab *pt_GPU, int M)
{
  int i;
	int u = threadIdx.x + 1;
	int m = threadIdx.x;
	int d = threadIdx.x - 1;
  float sig = sigmin + dsig*blockIdx.x;
	float mu = r - 0.5f*sig*sig;
	float pu = 0.25f*(sig*sig*dt/(dx*dx) + mu*dt/dx);
	float pm = 1.0f - 0.5*sig*sig*dt/(dx*dx);
	float pd = 0.25f*(sig*sig*dt/(dx*dx) - mu*dt/dx);
	float qu = -0.25f * (sig * sig * dt / (dx * dx) + mu * dt / dx);
	float qm = 1.0f + 0.5*sig * sig * dt / (dx * dx);
	float qd = -0.25f * (sig * sig * dt / (dx * dx) - mu * dt / dx);

	extern __shared__ float A[];

	float* sa = A;
	float* sd = sa + NTPB;
	float* sc = sd + NTPB;
	float* sy = sc + NTPB;
	int* sl = (int*)sy + 2*NTPB;

	sy[m] = pt_GPU[0][blockIdx.x][m];
	__syncthreads();

	for (i = 1; i <= (N/M) - 1; i++) {
		if (m == 0) {
      // the array is 2 times bigger
			sy[NTPB*(i%2) + m] = pmin;
		}
		else {
			if (m == NTPB - 1) {
				sy[NTPB*(i%2) + m] = pmax;
			}
			else {
        // Crank-N
				sy[NTPB*(i%2) + m] = pu*sy[NTPB * ((i+1) % 2) + u] + pm*sy[NTPB * ((i+1) % 2) + m] + pd*sy[NTPB * ((i+1) % 2) + d];
			}
		}
		sd[m] = qm;
		if (m < NTPB - 1) {
			sc[m + 1] = qu;
		}
		if (m > 0) {
			sa[m] = qd;
		}
		if (m == 0) {
			sa[0] = 0.f;
			sc[0] = 0.f;
		}
		sl[m] = m;

		__syncthreads();
		PCR_d(sa, sd, sc, sy + NTPB * (i % 2), sl, NTPB);
		__syncthreads();

		if (m == 0) {
			sy[NTPB * (i % 2)] = pmin;
			sy[NTPB * (i % 2) + NTPB - 1] = pmax;
		}
		__syncthreads();
	}

	pt_GPU[0][blockIdx.x][m] = sy[m+ NTPB*(N % 2)];
}

// Wrapper 
void PDE_diff (float dt, float dx, float dsig, float pmin, float pmax, float sigmin, int N, MyTab* CPUTab, int M)
{

	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions

	MyTab *GPUTab;
	testCUDA(cudaMalloc(&GPUTab, sizeof(MyTab)));
	
	testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice));
	// Accessing 2*N times to the global memory
	/*for(int i=0; i<N; i++){
	   PDE_diff_k1<<<NB,NTPB>>>(dt, dx, dsig, pmin, pmax, sigmin, GPUTab);
	}*/
	// Accessing only twice to the global memory
	//PDE_diff_k2<<<NB,NTPB,2*NTPB*sizeof(float)>>>(dt, dx, dsig, pmin, pmax, 
	//											  sigmin, N, GPUTab);
	//PDE_diff_k3<<<NB, NTPB, 5*NTPB*sizeof(float)>>>(dt, dx, dsig, pmin, pmax,
	//	sigmin, N, GPUTab);
	PDE_diff_k4<<<NB, NTPB, 6*NTPB*sizeof(float)>>>(dt, dx, dsig, pmin, pmax, sigmin, N, GPUTab, M);

	testCUDA(cudaMemcpy(CPUTab, GPUTab, sizeof(MyTab), cudaMemcpyDeviceToHost));

	testCUDA(cudaEventRecord(stop,0));				// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&TimeExec,start, stop));// GPU timer instructions
	testCUDA(cudaEventDestroy(start));				// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions

	testCUDA(cudaFree(GPUTab));	
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
	float K = 100.0f;
	float T = 1.0f;
	int N = 10000;
	float dt = (float)T/N;
	float xmin = log(0.5f*K);
	float xmax = log(2.0f*K);
	float dx = (xmax-xmin)/NTPB;
	float pmin = 0;
	float pmax = 2*K;
	float sigmin = 0.1f;
	float sigmax = 0.5f;
	float dsig = (sigmax-sigmin)/NB;
  float B = 120.0f;
  int P1 = 10, P2 = 50, M = 100;

	// Question 2
	MyTab *pt_CPU;
	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));
	for(int i=0; i<NB; i++){
	   for(int j=0; j<NTPB; j++){
      if (i >= P1 && i <= P2)
	      pt_CPU[0][i][j] = max(0.0, exp(xmin + dx*j) - K);
      else
        pt_CPU[0][i][j] = 0;
	   }
  }

	PDE_diff(dt, dx, dsig, pmin, pmax, sigmin, N, pt_CPU, M);

	testCUDA(cudaFreeHost(pt_CPU));
    // close files
    //fclose(file);
    return (0);
}