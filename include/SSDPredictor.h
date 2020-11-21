#pragma once
/*
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"
#include "BatchNormalization.h"
#include "Activation.h"
#include "NetStruct.h"
#include "Debug.h"
#include "Model.h"

void ReadWeights_(net* n)
{
	const char layer_name[][23] = {
		"_regression/", "_classification/"
	};
	const char weight_name[][23] = {
		"0_Conv","0_Bias","1_BatchNorm_mean","1_BatchNorm_var","1_BatchNorm_beta", "1_BatchNorm_gamma","3_Conv","3_Bias"};
	
	for( int li =0; li<12; li++)
	{
		layer* lptr = &n->layers[li];

		int idx = 0;
		for (int sli = 0; sli<lptr->size;sli++)
		{
			sublayer* slptr = &lptr->sublayers[sli];
			std::string target = "layer_"+std::to_string(li);
			if(li<6)
				target += layer_name[0];
			else
				target += layer_name[1];
			for(int opi = 0; opi<slptr->size;opi++)
			{
				operation* op = &slptr->ops[opi];
				if(op->opcode == CONV)
				{
					ReadBinFile_(&op->filter->data[0],target+weight_name[idx++]);
					ReadBinFile_(&op->bias->data[0],target+weight_name[idx++]);
				}
				else if(op->opcode == BN)
				{
					int offset = op->filter->w/4;
					for(int i=0;i<4;i++)
					{
						ReadBinFile_(&op->filter->data[i*offset],target+weight_name[idx++]);
					}
				}
			}
		}
	}
}




// Predictor is not an 1-way feed-forward Network
// but 5-way Netwark
net GetSSDPredictor()
{
	// regression(box locations)
	// classification(class scores)
	const int layer_sizes[] = {
		1,1,1,1,1,1,
		1,1,1,1,1,1
	};
	const int sublayer_sizes[] = {
		4,4,4,4,4,1,
		4,4,4,4,4,1
	};
	const int opcodes[] = {
		CONV,BN,RELU,CONV, CONV,BN,RELU,CONV, CONV,BN,RELU,CONV, CONV,BN,RELU,CONV, CONV,BN,RELU,CONV, CONV,
		CONV,BN,RELU,CONV, CONV,BN,RELU,CONV, CONV,BN,RELU,CONV, CONV,BN,RELU,CONV, CONV,BN,RELU,CONV, CONV
	};
	const int shapes[][4] = {
		{576,1,3,3},{1,1,1,576},{0,0,0,0},{24,576,1,1},
		{1280,1,3,3},{1,1,1,1280},{0,0,0,0},{24,1280,1,1}, 
		{512,1,3,3},{1,1,1,512},{0,0,0,0},{24,512,1,1}, 
		{256,1,3,3},{1,1,1,256},{0,0,0,0},{24,256,1,1},
		{256,1,3,3},{1,1,1,256},{0,0,0,0},{24,256,1,1}, 
		{64,64,1,1}, 
		
		{576,1,3,3},{1,1,1,576},{0,0,0,0},{126,576,1,1},
		{1280,1,3,3},{1,1,1,1280},{0,0,0,0},{126,1280,1,1}, 
		{512,1,3,3},{1,1,1,512},{0,0,0,0},{126,512,1,1}, 
		{256,1,3,3},{1,1,1,256},{0,0,0,0},{126,256,1,1},
		{256,1,3,3},{1,1,1,256},{0,0,0,0},{126,256,1,1}, 
		{64,64,1,1}, 
	};
	const int params[][3] = {
		{1,1,576},{0,0,0},{0,0,0},{1,0,1},
		{1,1,1280},{0,0,0},{0,0,0},{1,0,1},
		{1,1,512},{0,0,0},{0,0,0},{1,0,1},
		{1,1,256},{0,0,0},{0,0,0},{1,0,1},
		{1,1,256},{0,0,0},{0,0,0},{1,0,1},
		{1,0,1},
	
		{1,1,576},{0,0,0},{0,0,0},{1,0,1},
		{1,1,1280},{0,0,0},{0,0,0},{1,0,1},
		{1,1,512},{0,0,0},{0,0,0},{1,0,1},
		{1,1,256},{0,0,0},{0,0,0},{1,0,1},
		{1,1,256},{0,0,0},{0,0,0},{1,0,1},
		{1,0,1},
	};
	net predictor;
	net* nptr = &predictor;
	InitNetwork(nptr,"Predictor",12);
	
	int sublayer_i = 0;
	int opcode_i = 0;
	int shape_i = 0;
	int param_i = 0;
	
	for(int li=0;li<nptr->size;li++)
	{
		layer* lptr = &nptr->layers[li];
		InitLayer(lptr,layer_sizes[li],li);
		
		// Iterate layer's sublayers
		for(int sli = 0; sli < lptr->size; sli++)
		{
			sublayer* slptr = &lptr->sublayers[sli];
			InitSublayer(slptr,sublayer_sizes[sublayer_i++],sli);
			
			// Iterate sublayer's operation
			for(int opi = 0; opi < slptr->size; opi++)
			{
				operation* op = &slptr->ops[opi];

				op->filter = (Mat*)malloc(sizeof(Mat));
				op->bias  = (Mat*)malloc(sizeof(Mat));
				op->param = (Param*)malloc(sizeof(Param));

				InitMat(op->filter,shapes[shape_i++]);
				InitParam(op->param, params[param_i++]);
				InitOperation(op,opcodes[opcode_i++],opi);
			}
		}
	}
	ReadWeights_(&predictor);

	//PrintModel(&predictor);
	return predictor;	
}
*/
