#pragma once
#include <iostream>
#include <vector>
#include "Mat.h"
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
	int padding;
	int stride;
	int group;
	int dilation;
	Param(int padding_ ,int stride_, int group_)
		: padding(padding_), stride(stride_), group(group_)
	{
	}
	Param(const int (&shape)[3])
		: padding(shape[0]), stride(shape[1]), group(shape[2])
	{
	}
};

class Layer
{
public:
	int idx;
	int size;
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
	
	virtual void Forward();
	/*{
		std::cout<< "Layer" <<std::endl;
	}*/
};

class Sublayer : public Layer
{
public:
	Sublayer(int idx_ = -1, int size_ = -1)
		: Layer(idx_, size_)
	{
	}
	void Forward();
	/*{
		std::cout<< "Sublayer" <<std::endl;
	}*/
};


class Conv : public Layer
{
public:
	int opcode;
	Matrix* kernel;
	Matrix* bias;
	Param* param;
	Conv(Matrix* kernel_=NULL, Matrix* bias_=NULL, Param* param_=NULL)
		: kernel(kernel_), bias(bias_), param(param_)
	{
	}
	void Forward()
	{
		std::cout<< "CONV" <<std::endl;
	}
	Matrix Convolution_CPU(Matrix& input);
	Matrix Convolution_GPU(Matrix& input);
};

class BatchNorm : public Layer
{
public:
	int opcode;
	Matrix* mov_mean;
	Matrix* mov_var;
	Matrix* beta;
	Matrix* gamma;
	BatchNorm(Matrix* mov_mean_=NULL, Matrix* mov_var_=NULL, Matrix* beta_=NULL, Matrix* gamma_=NULL)
		: mov_mean(mov_mean_), mov_var(mov_var_), beta(beta_), gamma(gamma_)
	{
	}
	void Forward()
	{
		std::cout<< "BN" <<std::endl;
	}
	Matrix BatchNormalization_CPU(Matrix& input);
	Matrix BatchNormalization_GPU(Matrix& input);
};

class ReLU : public Layer
{
public:
	int opcode;
	int inplace;
	ReLU()
	{
	}
	void Forward()
	{
		std::cout<< "RELU" <<std::endl;
	}
	Matrix ReLU_CPU(Matrix& input);
	Matrix ReLU_GPU(Matrix& input);
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
	void Forward();
	void PrintNetwork();
};



