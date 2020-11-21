#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"
#include "Debug.h"


#define EPS 0.00001
// E -> moving-Mean
// Var -> moving-Variance
// 
// y = (gamma/( root( Var[x] + eps) ))*x + beta - gamma*E[x]/ ( root( Var[x] + eps ) ) 
// channel wise -> mean and var of batch*h*w
// filter ( 1, 1, 1, w)
// filter.data -> moving_mean, moving_var, beta, gamma
Mat* BatchNormalization(Mat* input, Mat* filter, Param* conv_p)
{
	const DTYPE eps = EPS;
	Mat* output = (Mat*)malloc(sizeof(Mat));
	InitMat(output, {input->n, input->c, input->h, input->w});
	
	int size = input->n*input->c;
	DTYPE* moving_mean = &filter->data[0];
	DTYPE* moving_var = &filter->data[1*size];
	DTYPE* beta = &filter->data[2*size];
	DTYPE* gamma = &filter->data[3*size];

	DTYPE* factorA = (DTYPE*)malloc(sizeof(DTYPE)*size);
	DTYPE* var = (DTYPE*)malloc(sizeof(DTYPE)*size);	
	//printf("filter shape(%d %d %d %d)\n",filter->n, filter->c, filter->h, filter->w);		
	for(int i = 0; i < size; i++)
	{
		var[i] = sqrt(moving_var[i] + eps);
		factorA[i] = gamma[i]/var[i];
		//printf("%f | %f | %f | %f\n",moving_mean[i], moving_var[i],beta[i],gamma[i]);
	}

	for(int oc = 0; oc<input->n; oc++)
    {
        for(int ic=0; ic<input->c; ic++)
        {
            for(int h=0; h<input->h; h++)
            {
                for(int w=0; w<input->w; w++)
                {
                    int data_index= oc*input->c*input->h*input->w 
						+ ic*input->h*input->w 
						+ h*input->w 
						+ w;
					int c_index = oc*input->c
						+ic;
                    output->data[data_index] = factorA[c_index]*(input->data[data_index] - moving_mean[c_index]) + beta[c_index];
                }
            }
        }
    }
	print("\t\t\t\tBatchNormalization\n");
	return output;
}


