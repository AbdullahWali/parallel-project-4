// CUDA assignment & Project 4 of Parallel Computing, CS426 at Bilkent University
// assignment details at http://cgds.me/cs426/project4.pdf
// Cagdas Oztekin 21201651
// Some inspiration from here: http://www.cs.usfca.edu/~peter/cs625/code/cuda-dot/dot3.cu
 
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// ---- HOST CODE

void fill_array(int* array, int size){
	time_t t;
	int i;

	srand((unsigned) time(&t));
	for(i = 0; i < size; i++){
		array[i] = rand() % size;
	}

	return;
}

// function to multiply two vectors of equal size in the CPU, two integer arrays and their size are passed to the function
int dot_product_cpu(int* first, int* second, int size){
   	int result = 0;
    int i;
    for(i = 0; i < size; i++){
            result += first[i] * second[i];
    }

    return result;
}

// ---- DEVICE CODE
__global__ void dot_product_gpu(const int* device_first, const int* device_second, int* device_size, int* device_num_threads, 
	int* result)
	// , int* MAX_BLOCKS)
{
	// find the thread id by first adding the thread id inside the block, then add the block offset by finding which block you are on
	// and also add grid offset considering our blocks have been laid out 3 dimensionally
	// so I suppose the structure is like:
	// threadIdx.x 							-> offset inside the block
	// blockIdx.x * blockDim.x 				-> offset of the block on its grid
	// blockIdx.y * blockDim.x * gridDim.x  -> which grid we are on times the size of a grid
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int id =  threadIdx.x;

	// can't figure out how to make this work, but the way they implemented at usfca didn't work
	// so I cheated more: http://stackoverflow.com/questions/9187899/cuda-shared-memory-array-variable
	extern __shared__ int sums[];

	// if our index is larger than the vector size that thread won't be doing any computation
	if (index < *device_size){
		sums[id] = device_first[index] * device_second[index];
	}
	else{
		sums[id] = 0;
	}
	__syncthreads();

	/*for (int stride = blockDim.x / 2; stride >  0; stride /= 2) {
		if (id < stride)
			sums[id] += sums[id + stride];
		__syncthreads();
	}*/

	// to ensure we're returning one sum from each block check the thread id, 0 is guaranteed to be there
	if (threadIdx.x == 0) {
		int i;
		int sum = 0;
		for(i = 0; i < *device_num_threads; i++){
			sum += sums[i];
		}

		// might lead to concurrency problems: TODO do some research about this
		atomicAdd(result, sum);	
	}	

	// retrieve the result from the device---
}

int main(int argc, char** argv)
{
	if(argc < 3){
		printf("usage: dot_product <array size> <number of threads per block>");
	}

	// take arguments from the user so that run tests quickly with a script
	// argv[1] is the size of our integer arrays
	const int SIZE = atoi(argv[1]);
	// argv[2] is the number of threads per block
	const int NUM_THREADS = atoi(argv[2]);

	int blocks = SIZE / NUM_THREADS;

	// pointers to the arrays in the host memory
	int* first = (int*)malloc(sizeof(int) * SIZE);
	int* second = (int*)malloc(sizeof(int) * SIZE);
	// pointers to the variables inside the host memory --necessary to copy them over to the device
	int* host_size = (int*)malloc(sizeof(int));
	int* host_num_threads = (int*)malloc(sizeof(int));
	// point the pointers to the variables
	*host_size = SIZE;
	*host_num_threads = NUM_THREADS;

	// results for the dot product in cpu and gpu will be held here
	int result_cpu = 0;
	// forgot to allocate this pointer in memory and took me like 20 minutes to find this mistake :-)
	int* result_gpu = (int*)malloc(sizeof(int));

	// variables to hold the runtimes for cpu & gpu
	float cpu_runtime = 0;
	float gpu_runtime = 0;
	float cuda_malloc_time = 0;
	float cuda_memcpy_time = 0;
	float cuda_calculation_time = 0;
	float cuda_memcpy_time_2 = 0;
	float array_creation = 0;

	// pointers for the arrays in the device
	int* device_first;
	int* device_second;
	// size and number of threads will also be copied to the device
	int* device_size;
	int* device_num_threads;
	int* device_result;
	int* device_max_blocks;

	printf("Info\n");
	printf("-----\n");
	printf("Number of elements: %d\n", SIZE);
	printf("Number of threads per block: %d\n", NUM_THREADS);
	printf("Number of blocks will be created: %d\n", SIZE / NUM_THREADS + 1);
	printf("Time\n");
	printf("-----\n");

	// timing variables, apparently clock() function I used to time the previous projects doesn't work out for CUDA
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// fill the two arrays
	cudaEventRecord(start, 0);
	fill_array(first, SIZE);
	fill_array(second, SIZE);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&array_creation, start, stop);

	printf("Time for the array generation : %.6f ms\n", array_creation);

	// time how long the dot product operation takes in the CPU
	cudaEventRecord(start, 0);
	result_cpu = dot_product_cpu(first, second, SIZE);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&cpu_runtime, start, stop);
	
	printf("Time for the CPU function: %.6f ms\n", cpu_runtime);

	// from now on it's for GPU
	cudaEventRecord(start, 0);
	// allocate memory in device for the data
	cudaMalloc((void**) &device_first, sizeof(int) * SIZE);
	cudaMalloc((void**) &device_second, sizeof(int) * SIZE);
	cudaMalloc((void**) &device_size, sizeof(int));
	cudaMalloc((void**) &device_num_threads, sizeof(int));
	cudaMalloc((void**) &device_result, sizeof(int));
	cudaMalloc((void**) &device_max_blocks, sizeof(int));
	cudaMemset(device_max_blocks, blocks, sizeof(int));
	cudaMemset(device_result, 0, sizeof(int));
	// finish timing for the allocations
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&cuda_malloc_time, start, stop);
	// printf("Memory allocation in device took: %fms\n", cuda_malloc_time);
	gpu_runtime += cuda_malloc_time;

	// copy the data in the host memory to device memory -- and don't forget to time the event
	cudaEventRecord(start, 0);
	cudaMemcpy(device_first, first, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(device_second, second, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(device_size, host_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_num_threads, host_num_threads, sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&cuda_memcpy_time, start, stop);
	gpu_runtime += cuda_memcpy_time;
	// REMARK: printing both the time for device memory allocation and data transfer
	printf("Time for the Host to Device transfer: %.6f ms\n", gpu_runtime);

	// TODO determine number of blocks and call the device function
	cudaEventRecord(start, 0);
	// add one to the number of blocks otherwise there is a risk some numbers may not be calculated if num_threads does not
	// divide size
	dot_product_gpu<<< SIZE / NUM_THREADS + 1, NUM_THREADS, NUM_THREADS * sizeof(int) >>>(device_first, device_second, device_size, 
		device_num_threads, device_result);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cuda_calculation_time, start, stop);


	printf("Time for the kernel execution: %.6f ms\n", cuda_calculation_time);
	gpu_runtime += cuda_calculation_time;

	cudaEventRecord(start, 0);
	cudaMemcpy(result_gpu, device_result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cuda_memcpy_time_2, start, stop);

	printf("Time for the Device to Host transfer: %fms\n", cuda_memcpy_time_2);

	gpu_runtime += cuda_memcpy_time_2;

	printf("Total execution time for Gpu: %.6f ms\n", gpu_runtime);

	// printf("GPU Result is: %d\n", *result_gpu);
	// printf("CPU Result is: %d\n", result_cpu);

	// printf("Runtime on CPU is: %fms\n", cpu_runtime);
	
	printf("Results\n");
	printf("-----\n");
	printf("Cpu result: %d\n", result_cpu);
	printf("Gpu result: %d\n", *result_gpu);



	// free memory allocations
	free(first);
	free(second);
	// free memory memory allocations in the device
	cudaFree(device_first);
	cudaFree(device_second);
	cudaFree(device_size);
	cudaFree(device_num_threads);
	cudaFree(device_result);

	return EXIT_SUCCESS;
}
