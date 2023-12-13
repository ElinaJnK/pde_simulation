#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define EPS 0.0000001f
#define NTPB 4
#define NB 50
#define P2 50
#define P1 10
#define r 0.1f
typedef float MyTab[P2+1][NTPB];

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

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

/*************************************************************************/
/*					Black-Sholes Formula								*/
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
void	print_arr(float *arr, int size)
{
	for(int i = 0; i < size; ++i)
		printf("%f ", arr[i]);
	printf("\n");
}

// Parallel cyclic reduction for implicit part
__device__ void PCR_d(float* sa, float* sd, float* sc,
    float* sy, int* sl, int n) {

    int i, lL, d, tL, tR;
    float aL, dL, cL, yL;
    float aLp, dLp, cLp, yLp;

    d = (n / 2 + (n % 2)) * (threadIdx.x % 2) + (int)threadIdx.x / 2;

    tL = threadIdx.x - 1;
    if (tL < 0) tL = 0;
    tR = threadIdx.x + 1;
    if (tR >= n) tR = 0;

    for (i = 0; i < (int)(logf((float)n) / logf(2.0f)) + 1; i++) {
        lL = (int)sl[threadIdx.x];

        aL = sa[threadIdx.x];
        dL = sd[threadIdx.x];
        cL = sc[threadIdx.x];
        yL = sy[threadIdx.x];

        dLp = sd[tL];
        cLp = sc[tL];

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

    sy[(int)sl[threadIdx.x]] = yL / dL;
}

/**
 * Question 2
 * PDE
*/
__global__ void PDE_diff_k4(float dt, float dx, float dsig, float pmin, float pmax, int N, MyTab *pt_GPU, MyTab *pt_GPU2, int M, int count)
{
	int i;
	int u = threadIdx.x + 1;
	int m = threadIdx.x;
	int d = threadIdx.x - 1;
	float sig = 0.2f;
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

	for (int w = 1; w <= (N/M) - 1; w++) 
	{
		i = w ;
		if (m == 0) {
	  		// the array is 2 times bigger
			sy[NTPB*(i%2) + m] = pmin;
		}
		else {
			if (m == NTPB - 1) {
				sy[NTPB*(i%2) + m] = pmax;
			}
			else {
				// Crank-Nicolson solution
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
	pt_GPU2[0][blockIdx.x][m] = sy[m+ NTPB*(N % 2)];
}

/**
 * Find the limit as specified on the discontinuities on T
*/
__global__ void find_lim(float xmin, float dx, float B, MyTab *GPUTab2, MyTab *GPUTab, int k)
{
	int		Pk_1;
	float	x;

	x = xmin + dx * threadIdx.x;
	Pk_1 = max(P1 - k, 0);
	if ((x < B) && (blockIdx.x == P2))
	{
		GPUTab[0][P2][threadIdx.x] = 0;
	}
	if (blockIdx.x == Pk_1 - 1)
	{
		if (x >=  B){
	  		GPUTab[0][Pk_1 - 1][threadIdx.x] = 0;
		}
		else {
	  		GPUTab[0][Pk_1 - 1][threadIdx.x] = GPUTab2[0][Pk_1][threadIdx.x] ;
	  	}
	}
	if (((blockIdx.x >= Pk_1) && (blockIdx.x < P2)))
	{
		if (x < B)
		{
			GPUTab[0][blockIdx.x][threadIdx.x] = GPUTab2[0][blockIdx.x + 1][threadIdx.x];
		}

	}
}


// Wrapper
/**
 * Question 2 requires to solve for the PDE on [Tm-1, T), so the loop is done
 * only once
*/
void PDE_diff (float dt, float dx, float dsig, float pmin, float pmax, int N, MyTab* CPUTab, int M, float K, float B)
{
	float TimeExec;									// GPU timer instructions
	cudaEvent_t start, stop;						// GPU timer instructions
	testCUDA(cudaEventCreate(&start));				// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));				// GPU timer instructions
	float xmin = log(K / 3);

	MyTab *GPUTab;
	MyTab *GPUTab2;
	testCUDA(cudaMalloc(&GPUTab, sizeof(MyTab)));
	testCUDA(cudaMalloc(&GPUTab2, sizeof(MyTab)));
	
	testCUDA(cudaMemcpy(GPUTab, CPUTab, sizeof(MyTab), cudaMemcpyHostToDevice));

	//testCUDA(cudaMemcpy(GPUTab2, CPUTab, sizeof(MyTab), cudaMemcpyDeviceToDevice));
	// Accessing 2*N times to the global memory
	/*for(int i=0; i<N; i++){
		PDE_diff_k1<<<NB,NTPB>>>(dt, dx, dsig, pmin, pmax, sigmin, GPUTab);
	}*/
	// Accessing only twice to the global memory
	//PDE_diff_k2<<<NB,NTPB,2*NTPB*sizeof(float)>>>(dt, dx, dsig, pmin, pmax, 
	//											  sigmin, N, GPUTab);
	//PDE_diff_k3<<<NB, NTPB, 5*NTPB*sizeof(float)>>>(dt, dx, dsig, pmin, pmax,
	//	sigmin, N, GPUTab);
	for (int i = 1; i <= 1; i++)
	{
		PDE_diff_k4<<<P2+1, NTPB, 6*NTPB*sizeof(float)>>>(dt, dx, dsig, pmin, pmax, N, GPUTab, GPUTab2, M, i);
		testCUDA(cudaDeviceSynchronize());
		// calculate the next limit
		find_lim<<<P2+1, NTPB>>>(xmin, dx, B, GPUTab2, GPUTab, i);
	}

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
	float xmin = log(K/3);
	float xmax = log(3*K);
	float dx = (xmax-xmin)/NTPB;
	float pmin = 0;
	float pmax = 2*K;
	float dsig = 0.2f;
	float B = 120.0f;
	int M = 100;

	MyTab *pt_CPU;
	testCUDA(cudaHostAlloc(&pt_CPU, sizeof(MyTab), cudaHostAllocDefault));
	for(int i=0; i<=P2; i++)
	{
		for(int j=0; j<NTPB; j++)
		{
			if (i >= P1 && i <= P2)
				pt_CPU[0][i][j] = max(0.0, exp(xmin + dx*j) - K);
			else
				pt_CPU[0][i][j] = 0;
		}
  	}


	PDE_diff(dt, dx, dsig, pmin, pmax, N, pt_CPU, M, K, B);
	for (int i = 0; i <= P2; i++)
	{
		for (int j = 0; j < NTPB; j++)
		{
	  		printf("%f\t", pt_CPU[0][i][j]);
		}
		printf("\n");
	}
	testCUDA(cudaFreeHost(pt_CPU));
	return (0);
}
