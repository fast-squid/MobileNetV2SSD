#pragma once
typedef float DTYPE;

class Matrix{
public:
	int n;
	int c;
	int h;
	int w;
	DTYPE* data;
	DTYPE* data_ptr;
	inline int Size()
	{
		return n*c*h*w;
	}

	Matrix(int n_, int c_, int h_, int w_)
		: n(n_), c(c_), h(h_), w(w_)
	{
	}
	Matrix(const int (&shape)[4])
		: n(shape[0]), c(shape[1]), h(shape[2]), w(shape[3])
	{
		data = (DTYPE*)malloc(sizeof(DTYPE)*Size());
	}

};
