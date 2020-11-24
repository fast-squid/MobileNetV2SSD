#include "network.h"
#include "convolution.cuh"

Matrix Network::Forward(Matrix& input)
{
	for(int i=0; i<size;i++)
	{
		inners[i]->Forward();
	}
}

Matrix Network::Forward(Matrix&input, int start, int end)
{	
	std::cout << name <<std::endl;
	for(int i=start; i<=end;i++)
	{
		inners[i]->Forward(input);
	}
}

Matrix Layer::Forward(Matrix& input)
{
	std::cout<< depth <<"layer : " << idx << std::endl;
	for(int i=0; i<size;i++)
	{
		if(inners[i]->opcode == CONV)
			((Conv*)inners[i])->Forward(input);
		else if(inners[i]->opcode == BN)
			((BatchNorm*)inners[i])->Forward(input);
		else if(inners[i]->opcode == RELU)
			((ReLU*)inners[i])->Forward(input);
		else
			inners[i]->Forward(input);
	}
}

/*void Sublayer::Forward()
{
	std::cout<< "sulayer" << idx<< std::endl;
	for(int i=0; i<size;i++)
	{
		inners[i]->Forward();
	}
}*/

Matrix Conv::Forward(Matrix& input)
{
	//Convolution_CPU();
	printf("Conv forward\n");
	return Convolution_GPU(input, *kernel, *bias, *param);
}


Matrix BatchNorm::Forward(Matrix& input)
{
	//BatchNormalization_CPU(input);	
}


Matrix ReLU::Forward(Matrix& input)
{

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
