#include "../include/network.h"
#include "../include/matrix.h"
#include <fstream>
#include <string>
void ReadBinFile_(DTYPE* data, std::string target)
{

    std::string root = "./Weights/";
    std::string test = "";
	test = root+target + ".bin";
    if( data==NULL)
    {
		printf("Data is NULL\n");
		exit(1);
	}

    int index=0;
    DTYPE load_val;
    std::ifstream read_file((string)test, std::ios::binary);
    if ( !read_file.is_open() )
    {
        std::cout<<"No Such Binaray"<<std::endl;
        exit(-1);
    }
    while( read_file.read(reinterpret_cast<char*>(&load_val), sizeof(DTYPE)))
    {
        data[index++] = load_val;
    }
}

void ReadWeights(Network& network)
{
	const char layer_name[][20] = {
		"_ConvBNRelu/", "_InvertedResidual/"
	};
	const char weight_name[][20] = {
		"0_Conv","1_BatchNorm_mean","1_BatchNorm_var","1_BatchNorm_beta", "1_BatchNorm_gamma",
		"3_Conv","4_BatchNorm_mean","4_BatchNorm_var","4_BatchNorm_beta", "4_BatchNorm_gamma",
		"6_Conv","7_BatchNorm_mean","7_BatchNorm_var","7_BatchNorm_beta", "7_BatchNorm_gamma",
	};
	
	for( int li =0; li<19; li++)
	{
		Layer* lptr = network.inners[li];

		int idx = 0;
		for (int sli = 0; sli<lptr->size;sli++)
		{
			Layer* sublayer = lptr->inners[sli];
			std::string target = "layer_"+std::to_string(li);
			if(li == 0 || li == 18)
				target += layer_name[0];
			else
				target += layer_name[1];
			for(int opi = 0; opi<sublayer->size;opi++)
			{
				Layer* operation_layer = sublayer->inners[opi];
				if(((Conv*)operation_layer)->opcode == CONV)
				{
					Conv* conv_layer = (Conv*)operation_layer;
					ReadBinFile_(&conv_layer->kernel->data[0], target+weight_name[idx++]);
				}
				else if(((BatchNorm*)operation_layer)->opcode == BN)
				{
					BatchNorm* bn_layer =(BatchNorm*) operation_layer;
					ReadBinFile_(&bn_layer->mov_mean->data[0],target+weight_name[idx++]);
					ReadBinFile_(&bn_layer->mov_var->data[0],target+weight_name[idx++]);
					ReadBinFile_(&bn_layer->beta->data[0],target+weight_name[idx++]);
					ReadBinFile_(&bn_layer->gamma->data[0],target+weight_name[idx++]);
				}
			}
		}
	}
}

void GetMobileNetV2(Network& network)
{
	const int network_size = 23;
	const int layer_sizes[] = {
		// base network
		1,3,4,4,4,
		4,4,4,4,4,
		4,4,4,4,4,
		4,4,4,1,	
		// extra network 
		4,4,4,4
	};

	const int sublayer_sizes[] = {
		// base network 
		3,																		// 0
		3,1,1,																	// 1
		3,3,1,1, 3,3,1,1, 3,3,1,1, 3,3,1,1,	3,3,1,1, 3,3,1,1, 3,3,1,1, 3,3,1,1,	// 2 ~
		3,3,1,1, 3,3,1,1, 3,3,1,1, 3,3,1,1, 3,3,1,1, 3,3,1,1, 3,3,1,1, 3,3,1,1,	// 17
		3,																		// 18
		// extra network 
		3,3,1,1, 3,3,1,1, 3,3,1,1, 3,3,1,1										
	};

	const int opcodes[] = {
		// base network 
		CONV,BN,RELU,							
		CONV,BN,RELU, CONV,BN,					
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU,
		// extra network
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN,	
		CONV,BN,RELU, CONV,BN,RELU, CONV,BN	
	};

	
	 //shapes[layer] = {n,c,h,w}
	const int shapes[][4] = {
		// base network
		{32,3,3,3},{1,1,1,32},    
		{32,1,3,3},{1,1,1,32},{16,32,1,1},{1,1,1,16},
		{96,16,1,1},{1,1,1,96},{96,1,3,3},{1,1,1,96},{24,96,1,1},{1,1,1,24},
		{144,24,1,1},{1,1,1,144},{144,1,3,3},{1,1,1,144},{24,144,1,1},{1,1,1,24},
		{144,24,1,1},{1,1,1,144},{144,1,3,3},{1,1,1,144},{32,144,1,1},{1,1,1,32},
		{192,32,1,1},{1,1,1,192},{192,1,3,3},{1,1,1,192},{32,192,1,1},{1,1,1,32},
		{192,32,1,1},{1,1,1,192},{192,1,3,3},{1,1,1,192},{32,192,1,1},{1,1,1,32},
		{192,32,1,1},{1,1,1,192},{192,1,3,3},{1,1,1,192},{64,192,1,1},{1,1,1,64},
		{384,64,1,1},{1,1,1,384},{384,1,3,3},{1,1,1,384},{64,384,1,1},{1,1,1,64},
		{384,64,1,1},{1,1,1,384},{384,1,3,3},{1,1,1,384},{64,384,1,1},{1,1,1,64},
		{384,64,1,1},{1,1,1,384},{384,1,3,3},{1,1,1,384},{64,384,1,1},{1,1,1,64},
		{384,64,1,1},{1,1,1,384},{384,1,3,3},{1,1,1,384},{96,384,1,1},{1,1,1,96},
		{576,96,1,1},{1,1,1,576},{576,1,3,3},{1,1,1,576},{96,576,1,1},{1,1,1,96},
		{576,96,1,1},{1,1,1,576},{576,1,3,3},{1,1,1,576},{96,576,1,1},{1,1,1,96},
		{576,96,1,1},{1,1,1,576},{576,1,3,3},{1,1,1,576},{160,576,1,1},{1,1,1,160},
		{960,160,1,1},{1,1,1,960},{960,1,3,3},{1,1,1,960},{160,960,1,1},{1,1,1,160},
		{960,160,1,1},{1,1,1,960},{960,1,3,3},{1,1,1,960},{160,960,1,1},{1,1,1,160},
		{960,160,1,1},{1,1,1,960},{960,1,3,3},{1,1,1,960},{320,960,1,1},{1,1,1,320},
		{1280,320,1,1},{1,1,1,1280},
		// extra network
		{256,1280,1,1},{1,1,1,256},{256,1,3,3},{1,1,1,256},{512,256,1,1},{1,1,1,512},
		{128,512,1,1},{1,1,128},{128,1,3,3},{1,1,1,128},{256,128,1,1},{1,1,1,256},
		{128,256,1,1},{1,1,1,128},{128,1,3,3},{1,1,1,128},{256,128,1,1},{1,1,1,256},
		{64,256,1,1},{1,1,1,64},{64,1,3,3},{1,1,1,64},{64,64,1,1},{1,1,1,64}
	};

	// param[layer] = {strides, padding, group}
	const int params[][3] = {
		// base network
		{2,1,1},
		{1,1,32},{1,0,1}, 
		{1,0,1},{2,1,96}, {1,0,1},
		{1,0,1},{1,1,144},{1,0,1},
		{1,0,1},{2,1,144},{1,0,1},
		{1,0,1},{1,1,192},{1,0,1},
		{1,0,1},{1,1,192},{1,0,1},
		{1,0,1},{2,1,192},{1,0,1},
		{1,0,1},{1,1,384},{1,0,1},
		{1,0,1},{1,1,384},{1,0,1},
		{1,0,1},{1,1,384},{1,0,1},
		{1,0,1},{1,1,384},{1,0,1},
		{1,0,1},{1,1,576},{1,0,1},
		{1,0,1},{1,1,576},{1,0,1},
		{1,0,1},{2,1,576},{1,0,1},
		{1,0,1},{1,1,960},{1,0,1},
		{1,0,1},{1,1,960},{1,0,1},
		{1,0,1},{1,1,960},{1,0,1},
		{1,0,1},
		// extra network
		{1,0,1},{2,1,256},{1,0,1},
		{1,0,1},{2,1,128},{1,0,1},
		{1,0,1},{2,1,128},{1,0,1},
		{1,0,1},{2,1,64}, {1,0,1}
	};
	network.size = network_size;
	network.name = "mobilenetv2";
	int sub_idx = 0;
	int opcode_idx = 0;
	int shape_idx = 0;
	int param_idx = 0;

	for(int layer_idx = 0; layer_idx < network.size; layer_idx++)
	{
		Layer* layer = new Layer(layer_idx, layer_sizes[layer_idx]);
		layer->opcode = -1;
		layer->depth = 1;
		for(int sublayer_idx = 0; sublayer_idx < layer->size; sublayer_idx++)
		{
			Layer* sublayer = new Layer(sublayer_idx,sublayer_sizes[sub_idx++]);
			sublayer->opcode = -1;
			sublayer->depth = 2;
			for(int operation_idx = 0; operation_idx < sublayer->size; operation_idx++)
			{
				if(opcodes[opcode_idx] == CONV)
				{
					Conv* conv = new Conv(new Matrix(shapes[shape_idx]), NULL, new Param(params[param_idx]));
					conv->opcode = opcodes[opcode_idx];
					conv->depth = 3;
					sublayer->PushInnerLayer(conv);
					shape_idx++; 
					param_idx++;
				}
				else if(opcodes[opcode_idx] == BN)
				{
					BatchNorm* bn = new BatchNorm(new Matrix(shapes[shape_idx]), new Matrix(shapes[shape_idx]), new Matrix(shapes[shape_idx]), new Matrix(shapes[shape_idx]));
					bn->opcode = opcodes[opcode_idx];
					bn->depth = 3;
					sublayer->PushInnerLayer(bn);
					shape_idx++;
				}
				else if(opcodes[opcode_idx] == RELU)
				{
					ReLU* relu = new ReLU();
					relu->opcode = opcodes[opcode_idx];
					sublayer->PushInnerLayer(relu);
				}
				opcode_idx++;
			}
			layer->PushInnerLayer(sublayer);
		}
		network.PushInnerLayer(layer);
	}
	ReadWeights(network);
}


