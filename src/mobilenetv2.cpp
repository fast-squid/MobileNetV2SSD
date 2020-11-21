#include <iostream>
#include <fstream>
#include <math.h>
#include <assert.h>
#include "network.h" 
#include "utils.h"

#include "Debug.h"
/*
void ReadBinFile(const char* path, DTYPE* data)
{
	int idx = 0;
	float temp;
	std::cout<< path<<std::endl;
	std::ifstream ifs(path, std::ios::binary);
    while( ifs.read(reinterpret_cast<char*>(&temp), sizeof(DTYPE)))
	{
		data[idx++] = temp;
	}
}

void CompareMat(Mat* x,Mat* y)
{
	int x_size = GetTotalSize(x);
	int y_size = GetTotalSize(y);
	bool flag = false;
	assert(x_size == y_size);
	
	for(int i = 0; i< x_size ; i++)
    {  
		//printf("%f | %f\n",x->data[i], y->data[i]);
		if( std::abs( x->data[i] - y->data[i] ) > 0.00002)
        {
            std::cout.precision(10);

            std::cout<<"ERROR "<< x->data[i]<<" | "<< y->data[i]<<std::endl;
            std::cout<<"IDX : "<<i<<std::endl;
			flag = true;
			break;
        }
    }
	if(flag) std::cout<<"Validation Failed" << std::endl;
	else std::cout<<"Validation Success" << std::endl;
}
*/

void Test()
{
	Network network;
	GetMobileNetV2(network);
	network.PrintNetwork();
	network.Forward();
}

int main()
{


	Test();
	return 0;
	/*
	Mat*input = (Mat*)malloc(sizeof(Mat));
	Mat*output,v_output, o[6];
	net mobilenetv2 = GetMobileNetV2SSD();
	
	InitMat(input, {1, 3, 224, 224});
	InitMat(&v_output,{1,1280,7,7});
	ReadBinFile("./Weights/input/input_data.bin",input->data);
	ReadBinFile("./Weights/layer_18_ConvBNRelu/imm_out.bin",v_output.data);

	printf("mobilenetv2 initialized\n");

	net predictor = GetSSDPredictor();
	printf("predictor initialized\n");

	// box predictor path
	SetDetourSublayerToLayer(&mobilenetv2.layers[14].sublayers[1], &predictor.layers[0]);	//14-1
	SetDetourLayerToLayer(&mobilenetv2.layers[18], &predictor.layers[1]);	//18
	SetDetourLayerToLayer(&mobilenetv2.layers[19], &predictor.layers[2]);	//ext0
	SetDetourLayerToLayer(&mobilenetv2.layers[20], &predictor.layers[3]);	//ext1
	SetDetourLayerToLayer(&mobilenetv2.layers[21], &predictor.layers[4]);	//ext2
	SetDetourLayerToLayer(&mobilenetv2.layers[22], &predictor.layers[5]);	//ext3

	// class predictor path
	SetDetourSublayerToLayer(&mobilenetv2.layers[14].sublayers[1], &predictor.layers[6]);	//14-1
	SetDetourLayerToLayer(&mobilenetv2.layers[18], &predictor.layers[7]);	//18
	SetDetourLayerToLayer(&mobilenetv2.layers[19], &predictor.layers[8]);	//ext0
	SetDetourLayerToLayer(&mobilenetv2.layers[20], &predictor.layers[9]);	//ext1
	SetDetourLayerToLayer(&mobilenetv2.layers[21], &predictor.layers[10]);	//ext2
	SetDetourLayerToLayer(&mobilenetv2.layers[22], &predictor.layers[11]);	//ext3
	
	output = Inference(&mobilenetv2,input,0,14);											// returns layer[13]'s output
	output = ForwardLayer(&mobilenetv2.layers[14], output, 0, 1);							// returns layer[14].sublayer[0]'s output
	CopyMat(&o[0], output);
	output = ForwardLayer(&mobilenetv2.layers[14], output, 1, mobilenetv2.layers[4].size);	// returns layer[14].sublayer[0]'s output
	
	output = Inference(&mobilenetv2, output, 18, 19);
	CopyMat(&o[0], output);
	output = Inference(&mobilenetv2, output, 19, 20);
	CopyMat(&o[2], output);
	output = Inference(&mobilenetv2, output, 20, 21);
	CopyMat(&o[3], output);
	output = Inference(&mobilenetv2, output, 21, 22);
	CopyMat(&o[4], output);
	output = Inference(&mobilenetv2, output, 22, 23);
	CopyMat(&o[5], output);

	output = Inference(&mobilenetv2, output, 18, 19);
	CopyMat(&o[0], output);
	output = Inference(&mobilenetv2, output, 19, 20);
	CopyMat(&o[2], output);
	output = Inference(&mobilenetv2, output, 20, 21);
	CopyMat(&o[3], output);
	output = Inference(&mobilenetv2, output, 21, 22);
	CopyMat(&o[4], output);
	output = Inference(&mobilenetv2, output, 22, 23);
	CopyMat(&o[5], output);
	
	// regression 
	ForwardLayer(&predictor.layers[0], &o[0], 0,1);
	ForwardLayer(&predictor.layers[1], &o[1], 0,1);
	ForwardLayer(&predictor.layers[2], &o[2], 0,1);
	ForwardLayer(&predictor.layers[3], &o[3], 0,1);
	ForwardLayer(&predictor.layers[4], &o[4], 0,1);
	ForwardLayer(&predictor.layers[5], &o[5], 0,1);

	// classification 
	ForwardLayer(&predictor.layers[6], &o[0], 0,1);
	ForwardLayer(&predictor.layers[7], &o[1], 0,1);
	ForwardLayer(&predictor.layers[8], &o[2], 0,1);
	ForwardLayer(&predictor.layers[9], &o[3], 0,1);
	ForwardLayer(&predictor.layers[10], &o[4], 0,1);
	ForwardLayer(&predictor.layers[511], &o[5], 0,1);







	//CompareMat(&v_output, output);

	return 0;
	*/
}


