#include "network.h"

void Network::Forward()
{
	for(int i=0; i<size;i++)
	{
		inners[i]->Forward();
	}
}

void Layer::Forward()
{
	std::cout<< idx << std::endl;
	for(int i=0; i<size;i++)
	{
		inners[i]->Forward();
	}
}

void Sublayer::Forward()
{
	std::cout<< idx<< std::endl;
	for(int i=0; i<size;i++)
	{
		inners[i]->Forward();
	}
}

void Conv::Forward()
{
	Convolution_CPU();
}

Matrix Conv::Convolution_CPU(const Matrix& input)
{	
}

Matrix Conv::Convolution_GPU(const Matrix& input)
{
}

void BatchNorm::Forward()
{
	BatchNormalization_CPU();	
}

Matrix BatchNormalization_CPU(const Matrix& input)
{
}

void ReLU::Forward()
{
}

Matrix ReLU::ReLU_CPU(Matrix& input)
{

	return 
}

void Network::PrintNetwork()
{
	std::cout<<name<<"{"<<std::endl;
	for(std::vector<Layer*>::iterator l = inners.begin() ; l!= inners.end(); l++)
	{
		std::cout<<"\tLayer" << (*l)->idx <<"{"<<std::endl;
		for(std::vector<Layer*>::iterator sl = (*l)->inners.begin(); sl!= (*l)->inners.end(); sl++)
		{
			std::cout<<"\t\tSublayer"<<(*sl)->idx<<"{"<<std::endl;
			for(std::vector<Layer*>::iterator op = (*sl)->inners.begin(); op!= (*sl)->inners.end(); op++)
			{
				if(((Conv*)(*op))->opcode == CONV)
				{
					std::cout << "\t\t\tConv"<<std::endl;
				}
				else if(((BatchNorm*)(*op))->opcode == BN)
				{
					std::cout <<"\t\t\tBatchNorm"<<std::endl;
				}
				else if(((ReLU*)(*op))->opcode == RELU)
				{
					std::cout <<"\t\t\tReLU"<<std::endl;
				}
			}
			std::cout<<"\t\t}"<<std::endl;
		}
		std::cout<<"\t}"<<std::endl;
	}
	std::cout<<"}"<<std::endl;
}
