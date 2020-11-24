#include "matrix.h"
#include "network.h"

typedef struct shape_
{
	int n;
	int c;
	int h;
	int w;
}shape;


__global__ void cudaConv2D(DTYPE* output, const shape o_shape,
		const DTYPE* input, const shape i_shape,
		const DTYPE* kernel, const shape k_shape,
		int stride, int padding, int group)
{
	int workload = o_shape.h * o_shape.w * i_shape.c;
	
	
	for(int tid = threadIdx.x; tid < workload ; tid += blockDim.x)
	{
		int input_c = (tid / (o_shape.h*o_shape.w));
		int input_h = (tid / o_shape.w)*stride;
		int input_w = (tid % o_shape.w)*stride;
		int input_idx = input_c*(i_shape.h*i_shape.w)
			+ input_h*(i_shape.w)
			+ input_w;
		for(int kh = 0 ; kh < k_shape.h; kh++)
		{
			for(int kc = 0; kc < k_shape.c; kc++)
			{
				for(int kw = 0; kw < k_shape.w; kw++)
				{
					int ker_idx = kc*(k_shape.h*k_shape.w)
						+ kh*(k_shape.w)
						+ kw;
					atomicAdd(&output[tid], input[input_idx] * kernel[ker_idx]);
				}
			}
		}
	}
}


void InitShape(shape& s, int n, int c, int h, int w)
{
	s.n = n;
	s.c = c;
	s.h = h;
	s.w = w;
}

Matrix Convolution_GPU(Matrix& input, Matrix& kernel, Matrix& bias, Param& p)
{
	Matrix output;
	output.n = 1;
	output.c = kernel.n;
	output.h = (input.h - kernel.h + 2*p.padding)/p.stride + 1;
	output.w = (input.w - kernel.w + 2*p.padding)/p.stride + 1;
	output.data = new DTYPE[output.Size()];

	DTYPE* output_d;
	DTYPE* input_d;
	DTYPE* kernel_d;
	
	cudaMalloc((void**)&output_d, sizeof(DTYPE)*output.Size());
	cudaMalloc((void**)&input_d, sizeof(DTYPE)*input.Size());
	cudaMalloc((void**)&kernel_d, sizeof(DTYPE)*kernel.Size());

	cudaMemset(output_d, 0, sizeof(DTYPE)*output.Size());
	cudaMemcpy(input_d, input.data, sizeof(DTYPE)*input.Size(), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_d, kernel.data, sizeof(DTYPE)*kernel.Size(), cudaMemcpyHostToDevice);
	
	// # of blocks = kernel.n
	// one output channel is produced by oh*ow*ic threads
	int block_num = kernel.n;
	int thread_num = 1024;
	
	shape o_shape;	
	shape i_shape;
	shape k_shape;
	InitShape(o_shape, output.n,output.c, output.h, output.w);
	InitShape(i_shape, input.n, input.c, input.h, input.w);
	InitShape(k_shape, kernel.n,kernel.c, kernel.h, kernel.w);

	cudaConv2D<<<block_num, thread_num>>>(
			output_d, o_shape,
			input_d, i_shape,
			kernel_d, k_shape,
			p.padding, p.stride, p.groups);

	return output;


}
