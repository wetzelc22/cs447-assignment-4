#include <cuda.h>
#include <cmath>
#include <thread>
#include <random>
#include <vector>
#include <iostream>
#include <memory>
#include <limits>

__global__ void
function(int n, int n_threads, double step, double * sum){
	double temp = (step * n) / n_threads;
	double start = temp * threadIdx.x;
	double x = start;
	double sum_part = 0.0;

	double n_per = n / n_threads;

	for(int i = 0; i < n_per; i++){
		if(x != 0.0){
			double val = sin(x)/x;
			sum_part += val;
		}
		x += step;
	}

	atomicAdd(sum, sum_part);

	__syncthreads();

}

int main(int argc, char **argv){

	cudaError_t rv;

	rv = cudaDeviceReset();
	assert(rv == cudaSuccess);

	if(argc != 5) {std::cerr<< "Incorrect number of arguments" << std::endl; return EINVAL;};
	
	float a = std::stod(argv[1]);
	float b = std::stod(argv[2]);
	int n = atoi(argv[3]);
	int n_threads = std::stoull(argv[4]);
	if(n_threads < 1) {std::cerr << "Incorrect number of arguments" << std::endl; return EINVAL;};
	//Here the number of steps per thread is calculated and the size of each subsection is also calculated and set to temp
	int n_per = n / n_threads;
	float temp  = std::abs((b-a)) / n_threads;
	float y = a;
	double step = (b-a)/n;

	//create sum on global mem
	double * sum;
	rv = cudaMalloc(&sum, sizeof(double));
	assert(rv == cudaSuccess);
	double *sum_temp = (double *)malloc(sizeof(double));
	*sum_temp = 0;
	cudaMemcpy(sum, sum_temp, sizeof(double), cudaMemcpyHostToDevice);
	//and have it set to 0
	//cuda kernel call
	function<<<1, n_threads>>>(n, n_threads, step, sum);

	//Here the different values of the trapezoidal rule are calculated to give the result as "answer"
	double val2 = 0.0; 
	if(a != 0) val2 = (sin(a)/a);
	double val3 = 0.0;
	if(b != 0) val3 = (sin(b)/b);
	val3 = val3 / 2;
	val2 = val2 / 2;
	typedef std::numeric_limits< double > dbl;
	std::cout.precision(dbl::max_digits10);
	double answer = step * (val2 + sum + val3);
	std::cout << answer << std::endl;

	rv = cudaFree(sum);
	assert(rv == cudaSuccess);
	free(sum_temp);

	return 0;

}
