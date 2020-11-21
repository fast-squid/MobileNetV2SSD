#pragma once
#include <cmath>
#include <iostream>
#include "DataStruct.h"
#include <string.h>
#include "Debug.h"

void PaddingInputImage(const Mat* p_input, int pad ,Mat* pad_temp)
{
    int n = p_input->n;
    int c = p_input->c;
    int pad_h = p_input->h + 2*pad;
    int pad_w = p_input->w + 2*pad;
	//InitMat(p_input, {n, c, pad_h, pad_w});

	int total_allocation_size = p_input->n * p_input->c * (p_input->h+2*pad) * (p_input->w+2*pad);

    pad_temp->data = (DTYPE*)malloc(sizeof(DTYPE)*total_allocation_size);
    pad_temp->n=p_input->n;
    pad_temp->c=p_input->c;
    pad_temp->h=p_input->h+2*pad;
    pad_temp->w=p_input->w+2*pad;

    for( int i0 =0; i0< n; i0++)
    {   
        for(int i1 =0; i1< c; i1++)
        {
            for(int i2=0; i2< pad_h; i2++)
            {
                for(int i3=0; i3< pad_w; i3++)
                {

                        int pad_index = i0*c*pad_h*pad_w
                                        +i1*pad_h*pad_w
                                        +i2*pad_w
                                        +i3;
                        int input_index = i0*c*p_input->h*p_input->w
                                        + i1*p_input->h*p_input->w
                                        + i2* p_input->w
                                        + i3 -(p_input->w*pad+pad);

                    if( ((pad <= i2)&&(i2 < pad_h-pad)) && ((pad <= i3)&&(i3 < pad_w-pad)) )
                    {
                        pad_temp->data[pad_index]= p_input->data[ input_index ];
                    }
                    else
                    {
                        pad_temp->data[pad_index]= 0.0f;
                    }
                }
            }
        }
    }
    return;
}

void SetOutputShape(Mat* input, Mat* filter, Mat* output, Param* conv_p)
{
	output->n = 1;
	output->c = filter->n;
	output->h = floor( (DTYPE)(input->h - filter->h +2*conv_p->padding)/conv_p->strides +1);
	output->w = floor( (DTYPE)(input->w - filter->w +2*conv_p->padding)/conv_p->strides +1 );
	//output->data = (DTYPE*)malloc(sizeof(DTYPE)*output->n*output->c*output->h*output->w); 
}

Mat* Convolution(Mat* input, Mat* filter, Param* conv_p )
{
	int groups = conv_p->groups;
	
	// init output
	Mat* output = (Mat*)malloc(sizeof(Mat));
	SetOutputShape(input, filter, output, conv_p);
	InitMat(output, {output->n, output->c, output->h, output->w});
	// splitting by groups
	Mat sliced_input = *input;
	Mat sliced_output = *output;
	Mat sliced_filter = *filter;

	sliced_input.c/=groups;
	sliced_filter.n/=groups; // # of filters
	sliced_output.c = sliced_filter.n;

	int in_offset = sliced_input.c
		*sliced_input.h
		*sliced_input.w;
	int out_offset = sliced_output.c
		*sliced_output.h
		*sliced_output.w;	
	int filter_offset = sliced_filter.c
		*sliced_filter.n
		*sliced_filter.h
		*sliced_filter.w;
	
	for(int g = 0; g<groups; g++)
	{
		sliced_input.data = &input->data[g*in_offset];
		sliced_output.data = &output->data[g*out_offset];
		sliced_filter.data = &filter->data[g*filter_offset];
		
		Mat pad_input;
		PaddingInputImage(&sliced_input, conv_p->padding, &pad_input);

		
		// ic,kh,kw ---> reduction index
		// output_d += Pad_input[ic][oh+kh][ow+hw]*Filter[ic][kh][kw]
		for(int oc=0; oc< sliced_output.c; oc++ )
		{
			for(int oh=0; oh<sliced_output.h; oh++)
			{
				for( int ow=0; ow<sliced_output.w; ow++)
				{
					int out_index = oc*sliced_output.h*sliced_output.w
						+ oh*sliced_output.w
						+ ow;
					sliced_output.data[out_index] = 0;

					/// Reduction Phase
					for( int ic=0; ic< sliced_filter.c; ic++)
					{
						for( int kh=0; kh<sliced_filter.h; kh++)
						{
							for( int kw=0; kw<sliced_filter.w; kw++)
							{
								int pad_index = ic*pad_input.h*pad_input.w
									+ oh*(conv_p->strides)*pad_input.h + kh*pad_input.h
									+ ow*(conv_p->strides) + kw;

								int kernel_index = oc*sliced_filter.c*sliced_filter.h*sliced_filter.w
									+ ic*sliced_filter.h*sliced_filter.w
									+ kh*sliced_filter.w
									+ kw;
								sliced_output.data[out_index] +=  sliced_filter.data[kernel_index]* pad_input.data[pad_index] ;
							}
						}
					}
				}
			}
		}
		free( pad_input.data );

	}
	print("\t\t\t\tConv\n");
    return output;
}


