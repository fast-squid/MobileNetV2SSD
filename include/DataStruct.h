#pragma once
#include "Debug.h"
/*
inline int GetTotalSize(Mat* mat)
{
	return mat->n * mat->c * mat->h * mat->w; 
}

inline int GetMatSize(Mat* mat)
{
	return mat->h * mat->w; 
}

void InitMat(Mat* mat, const int (&shape)[4])
{
    mat->n = shape[0];
    mat->c = shape[1];
    mat->h = shape[2];
    mat->w = shape[3];
	int total_size =GetTotalSize(mat);
	
	if(total_size)
	    mat->data = (DTYPE*)malloc(sizeof(DTYPE)*total_size);
	else 
		free(mat);
}

void PrintMat(Mat* mat)
{
	printf("shape(%d,%d,%d,%d)\n", mat->n, mat->c, mat->h, mat->w);
}

void CopyMat(Mat* src,Mat* dst)
{
	src-> n = dst->n;
	src-> c = dst->c;
	src-> h = dst->h;
	src-> w = dst->w;
	src->data = (DTYPE*)malloc(sizeof(DTYPE)*GetTotalSize(src));
	for(int i = 0; i < GetTotalSize(src); i++)
	{
		src->data[i] = dst->data[i];
	}

}
void FreeMat(Mat* mat)
{
	if(mat->data)
	{
		print("free!\n");
		free(mat->data);
		free(mat);
	}
	else{
		print("can't free");
	}
}

Mat* PermuteMat(Mat* mat, const int shape[4])
{	
	DTYPE* new_data = (DTYPE*)malloc(sizeof(DTYPE)*GetTotalSize(mat));
	//TODO: general Permute
	int idx = 0;
	int n = mat->n;
	int c = mat->c;
	int h = mat->h;
	int w = mat->w;

	for(int ni = 0; ni < n; ni++)
	{
		for(int hi = 0; hi < h; hi++)
		{
			for(int wi = 0; wi < w; wi++)
			{
				for(int ci = 0; ci < c; ci++)
				{
					int org_idx = ni*(h*w*c)
						+ ci*(h*w)
						+ hi*w
						+ wi;
					new_data[idx++] = mat->data[org_idx];
				}
			}
		}
	}
	free(mat->data);
	mat->data = new_data;
	return mat;
}

void InitParam(Param* conv_p, const int (&param)[3])
{
    conv_p->strides = param[0];
	conv_p->padding = param[1];
    conv_p->groups = param[2];
}


*/
