#pragma once
#include <iostream>
#include <vector>
#include "matrix.h"
#include "Debug.h"

const int CONV = 0;
const int BN = 1;
const int RELU = 2;
const int DROP = 3;
const int LINEAR = 4;
const int NETSIZE = 5;

using namespace std;

class Param
{
public:	
	int stride;
	int padding;
	int groups;
	int dilation;
	Param(int stride_ ,int padding_, int groups_)
		: stride(stride_), padding(padding_), groups(groups_)
	{
	}
	Param(const int (&shape)[3])
		: stride(shape[0]), padding(shape[1]), groups(shape[2])
	{
	}
};

class Layer
{
public:
	int idx;
	int size;
	int depth;
	int opcode;
	vector<Layer*> inners;
	Layer()
	{
	}
	Layer(int idx_, int size_)
		: idx(idx_), size(size_)
	{
	}
	virtual void PushInnerLayer(Layer* l)
	{
		inners.push_back(l);
	}
	
	virtual Matrix Forward(){};
	virtual Matrix Forward(Matrix& input);

};


class Conv : public Layer
{
public:
	Matrix* kernel;
	Matrix* bias;
	Param* param;
	Conv(Matrix* kernel_=NULL, Matrix* bias_=NULL, Param* param_=NULL)
		: kernel(kernel_), bias(bias_), param(param_)
	{
	}
	Matrix Forward(Matrix& input);
};

class BatchNorm : public Layer
{
public:
	Matrix* mov_mean;
	Matrix* mov_var;
	Matrix* beta;
	Matrix* gamma;
	BatchNorm(Matrix* mov_mean_=NULL, Matrix* mov_var_=NULL, Matrix* beta_=NULL, Matrix* gamma_=NULL)
		: mov_mean(mov_mean_), mov_var(mov_var_), beta(beta_), gamma(gamma_)
	{
	}
	Matrix Forward(Matrix& input);
};

class ReLU : public Layer
{
public:
	int inplace;
	ReLU()
	{
	}
	Matrix Forward(Matrix& input);
};

class Network : public Layer
{
public:
	string name;
	Network(string str="") : name(str)
	{
	}
	~Network()
	{
		
	}
	Matrix Forward(Matrix& input);
	Matrix Forward(Matrix& input, int start, int end);
	void PrintNetwork();
};



