#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include "Convolution.h"
#include "Debug.h"
Mat* Relu6(Mat* input_data, Mat* filter, Param* conv_p)
{
	Mat* output;

    for(int i0=0; i0< input_data->n; i0++)
    {
        for(int i1=0; i1<input_data->c; i1++)
        {
            for(int i2=0; i2<input_data->h; i2++)
            {
                for(int i3=0; i3<input_data->w; i3++)
                {
                    int i_index= i0*input_data->c*input_data->h*input_data->w
                                    + i1 * input_data->h*input_data->w
                                    + i2 * input_data->w
                                    + i3;

                    if( input_data->data[i_index] <= 0.0f )
                    {
                        input_data->data[i_index]=0;
                    }
                    else if( input_data->data[i_index] >= 6.0f )
                    {
                        input_data->data[i_index] = 6;
                    }
                }
            }
        }
    }
	print("\t\t\t\tReLU6\n");
	output = input_data;
	return output;
}
