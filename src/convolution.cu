typedef struct shape_{
	int n;
	int c;
	int h;
	int w;
}shape;

__global__ ConvolutionNCHW(DTYPE* output, shape o_shape,
		const DTYPE* input, const shape i_shape,
		const DTYPE* kernel, const shape k_shape,
		int stride, int padding, int group)
{
	int tid;
	int workload = o_shape.h * o_shape.w;
	
	__shared__ float s_input[1024];
	int output_channel = blockIdx.x;
	int i_row = threadIdx.x/o_shape.w; // example 1024 / 150 == 6
	int o_idx = threadIdx.x;

	for(int i = 0; i < workload / blockDim.x; i++)
	{
		int row = o_idx/o_shape.w;
		int col = o_idx%o_shape.w;
		int o_idx = 
		for(int kh = threadIdx.x ; kh < k_shape.h; kh++)
		{
			// load single row of input
			// tid   0 ~ 149 : s_input[tid] = input[0*300 + tid]
			// tid 150 ~ 299 : s_input[tid] = 
			for(int tid = threadIdx.x; tid < row*i_shape.w; tid += o_shape.w)
			{
				s_input[tid] = input[row*i_shape.w + tid];
			}

			__syncthreads();

			int in_w_base = threadIdx.x*stride;
			for(int kc = 0; kc < k_shape.c; kc++)
			{
				for(int kw = 0; kw < k_shape.w; kw++)
				{
					output[o_idx] = s_input[in_w_base + kw] * kernel[kw];
				}
			}
		}

		o_idx += blockDim.x;
	}

	
}




Matrix Convolution_GPU(Matrix& input, Matrix& kernel, Matrix& bias)
{
	// # of blocks = kernel.n
	// one output channel is produced by oh*ow threads

}
