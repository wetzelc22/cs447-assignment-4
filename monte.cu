#include <cuda.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <limits>
#include <cassert>

__global__ void
function(float a, int n_per,  double step, float temp, double * sum_array){
	double sum_part = 0.0;
	double x = a + (temp * (threadIdx.x));
	for(int i = 0; i < n_per; i++){
		if(x != 0.0){
			double val = sin(x)/x;
			sum_part += val;
		}
		x += step;
	}

	sum_array[threadIdx.x] = sum_part;

	//__syncthreads();

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
	float temp  = std::abs((b-a)) / n_threads;
	double step = (b-a)/n;
	int n_per = n / n_threads;
	//create sum on global mem
	double sum = 0.0;
	double *sum_array;
	rv = cudaMalloc(&sum_array, n_threads * sizeof(double));
	assert(rv == cudaSuccess);
	double *sum_temp = (double *)malloc(n_threads * sizeof(double));
	for(int i = 0; i < n_threads; i++){
		sum_temp[i] = 0.0;
	}
	cudaMemcpy(sum_array, sum_temp, n_threads * sizeof(double), cudaMemcpyHostToDevice);
	//and have it set to 0
	//cuda kernel call
	function<<<1, n_threads>>>(a, n_per, step, temp, sum_array);
	cudaMemcpy(sum_temp, sum_array, n_threads * sizeof(double), cudaMemcpyDeviceToHost);
	for(int i = 0; i < n_threads; i++){
		sum += sum_temp[i];
	}
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

	rv = cudaFree(sum_array);
	assert(rv == cudaSuccess);
	free(sum_temp);

	return 0;

}
