#pragma once
#include <stdio.h>
#include <string.h>
typedef float DTYPE;

class Matrix{
public:
	int n;
	int c;
	int h;
	int w;
	DTYPE* data;
	DTYPE* data_ptr;

	int Size() const
	{
		return n*c*h*w;
	}
	Matrix(){}
	Matrix(int n_, int c_, int h_, int w_)
		: n(n_), c(c_), h(h_), w(w_)
	{
		data = (DTYPE*)calloc(Size(),sizeof(DTYPE));
	}
	Matrix(const int (&shape)[4])
		: n(shape[0]), c(shape[1]), h(shape[2]), w(shape[3])
	{
		data = (DTYPE*)calloc(Size(),sizeof(DTYPE));
	}
	void PrintShape()
	{ 
		printf("(n,c,h,w) : (%d, %d, %d, %d) matrix[%d]:%f, matrix[%d]:%f\n",n,c,h,w,0,data[0],Size()-1, data[Size()-1] );
	}
	Matrix PadMatrix(int padding)
	{
		Matrix p_input(n, c, h+2*padding, w+2*padding);
		for(int ic = 0; ic < p_input.c; ic++)
		{
			for(int ih = 0; ih < p_input.h; ih++)
			{
				if(ih < padding || ih > ((p_input.h-1)-padding))
					continue;
				else
				{ 
					int p_idx = ic*(p_input.h*p_input.w)
						+ ih*p_input.w
						+ padding;
					int i_idx = ic*(h*w)
						+ (ih-padding)*w;
					memcpy(&p_input.data[p_idx], &data[i_idx], sizeof(DTYPE)*w);
				}
			}
		}
		return p_input;
	}


};
