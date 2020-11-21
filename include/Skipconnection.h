#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include <string.h>

Mat* MatAdd(Mat* input, Mat* output)
{
	Mat* new_output = (Mat*)malloc(sizeof(Mat));
	InitMat(new_output,{input->n, input->c, input->h, input->w});
	for(int i=0;i<GetTotalSize(input);i++)
	{
		new_output->data[i] =  output->data[i] + input->data[i];
	}
	print("\tMat Add\n");
    return new_output;
}


