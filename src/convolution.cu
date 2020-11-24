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
	int workload = o_shape.h * o_shape.w;
	int blk_oidx = blockIdx.x *(o_shape.w*o_shape.h);
	int blk_kidx = blockIdx.x *(k_shape.c*k_shape.w*k_shape.h);
	for(int c = 0 ; c<i_shape.c;c++)
	{	
		for(int tid = threadIdx.x; tid < workload ; tid += blockDim.x)
		{
			int in_w = (tid%o_shape.w)*stride;
			int in_h = (tid/o_shape.w)*stride;
			int input_idx = c*i_shape.h*i_shape.w
				+ in_h*i_shape.w
				+ in_w;
			for(int kh = 0 ; kh < k_shape.h; kh++)
			{
				for(int kw = 0; kw < k_shape.w; kw++)
				{
					int ker_idx = c*(k_shape.h*k_shape.w)
						+ kh*(k_shape.w)
						+ kw;
					atomicAdd(&output[blk_oidx+tid], input[input_idx+(kh*i_shape.w+kw)]
							* kernel[blk_kidx+ker_idx]);
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
	Matrix pad_input = input.PadMatrix(p.padding);
	
	int out_h = (pad_input.h-kernel.h)/p.stride+1;
	int out_w = (pad_input.w-kernel.w)/p.stride+1;
	Matrix output(input.n, kernel.n, out_h, out_w);	
	DTYPE* output_d;
	DTYPE* input_d;
	DTYPE* kernel_d;
	
	cudaMalloc((void**)&output_d, sizeof(DTYPE)*output.Size());
	cudaMalloc((void**)&input_d, sizeof(DTYPE)*pad_input.Size());
	cudaMalloc((void**)&kernel_d, sizeof(DTYPE)*kernel.Size());

	//cudaMemset(output_d, 0, sizeof(DTYPE)*output.Size());
	cudaMemcpy(input_d, pad_input.data, sizeof(DTYPE)*pad_input.Size(), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_d, kernel.data, sizeof(DTYPE)*kernel.Size(), cudaMemcpyHostToDevice);
	
	// # of blocks = kernel.n
	// one output channel is produced by oh*ow*ic threads
	int block_num = kernel.n;
	int thread_num = 1024;
	
	shape o_shape;	
	shape i_shape;
	shape k_shape;
	InitShape(o_shape, output.n,output.c, output.h, output.w);
	InitShape(i_shape, pad_input.n, pad_input.c, pad_input.h, pad_input.w);
	InitShape(k_shape, kernel.n,kernel.c, kernel.h, kernel.w);
	
	cudaConv2D<<<block_num, thread_num>>>(
			output_d, o_shape,
			input_d, i_shape,
			kernel_d, k_shape,
			p.stride, p.padding, p.groups);

	cudaMemcpy(output.data, output_d, sizeof(DTYPE)*output.Size(),cudaMemcpyDeviceToHost);
	cudaFree(input_d);
	cudaFree(output_d);
	cudaFree(kernel_d);

	free(pad_input.data);
	return output;
}
