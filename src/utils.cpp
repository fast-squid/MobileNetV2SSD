#include "../include/network.h"
#include "../include/Mat.h"

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
		{32,3,3,3},{1,1,1,128},    
		{32,1,3,3},{1,1,1,128},{16,32,1,1},{1,1,1,64},
		{96,16,1,1},{1,1,1,384},{96,1,3,3},{1,1,1,384},{24,96,1,1},{1,1,1,96},
		{144,24,1,1},{1,1,1,576},{144,1,3,3},{1,1,1,576},{24,144,1,1},{1,1,1,96},
		{144,24,1,1},{1,1,1,576},{144,1,3,3},{1,1,1,576},{32,144,1,1},{1,1,1,128},
		{192,32,1,1},{1,1,1,768},{192,1,3,3},{1,1,1,768},{32,192,1,1},{1,1,1,128},
		{192,32,1,1},{1,1,1,768},{192,1,3,3},{1,1,1,768},{32,192,1,1},{1,1,1,128},
		{192,32,1,1},{1,1,1,768},{192,1,3,3},{1,1,1,768},{64,192,1,1},{1,1,1,256},
		{384,64,1,1},{1,1,1,1536},{384,1,3,3},{1,1,1,1536},{64,384,1,1},{1,1,1,256},
		{384,64,1,1},{1,1,1,1536},{384,1,3,3},{1,1,1,1536},{64,384,1,1},{1,1,1,256},
		{384,64,1,1},{1,1,1,1536},{384,1,3,3},{1,1,1,1536},{64,384,1,1},{1,1,1,256},
		{384,64,1,1},{1,1,1,1536},{384,1,3,3},{1,1,1,1536},{96,384,1,1},{1,1,1,384},
		{576,96,1,1},{1,1,1,2304},{576,1,3,3},{1,1,1,2304},{96,576,1,1},{1,1,1,384},
		{576,96,1,1},{1,1,1,2304},{576,1,3,3},{1,1,1,2304},{96,576,1,1},{1,1,1,384},
		{576,96,1,1},{1,1,1,2304},{576,1,3,3},{1,1,1,2304},{160,576,1,1},{1,1,1,640},
		{960,160,1,1},{1,1,1,3840},{960,1,3,3},{1,1,1,3840},{160,960,1,1},{1,1,1,640},
		{960,160,1,1},{1,1,1,3840},{960,1,3,3},{1,1,1,3840},{160,960,1,1},{1,1,1,640},
		{960,160,1,1},{1,1,1,3840},{960,1,3,3},{1,1,1,3840},{320,960,1,1},{1,1,1,1280},
		{1280,320,1,1},{1,1,1,5120},
		// extra network
		{256,1280,1,1},{1,1,1,1024},{256,1,3,3},{1,1,1,1024},{512,256,1,1},{1,1,1,2048},
		{128,512,1,1},{1,1,1,512},{128,1,3,3},{1,1,1,512},{256,128,1,1},{1,1,1,1024},
		{128,256,1,1},{1,1,1,512},{128,1,3,3},{1,1,1,512},{256,128,1,1},{1,1,1,1024},
		{64,256,1,1},{1,1,1,256},{64,1,3,3},{1,1,1,256},{64,64,1,1},{1,1,1,256}
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
	int sub_idx = 0;
	int opcode_idx = 0;
	int shape_idx = 0;
	int param_idx = 0;

	for(int layer_idx = 0; layer_idx < network.size; layer_idx++)
	{
		Layer* layer = new Layer(layer_idx, layer_sizes[layer_idx]);
		for(int sublayer_idx = 0; sublayer_idx < layer->size; sublayer_idx++)
		{
			Sublayer* sublayer = new Sublayer(sublayer_idx,sublayer_sizes[sub_idx++]);
			for(int operation_idx = 0; operation_idx < sublayer->size; operation_idx++)
			{
				if(opcodes[opcode_idx] == CONV)
				{
					Conv* conv = new Conv(new Matrix(shapes[shape_idx]), NULL, new Param(params[param_idx]));
					conv->opcode = opcodes[opcode_idx];
					sublayer->PushInnerLayer(conv);
					shape_idx++; 
					param_idx++;
				}
				else if(opcodes[opcode_idx] == BN)
				{
					BatchNorm* bn = new BatchNorm(new Matrix(shapes[shape_idx]), new Matrix(shapes[shape_idx]), new Matrix(shapes[shape_idx]), new Matrix(shapes[shape_idx]));
					bn->opcode = opcodes[opcode_idx];
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

}

void ReadBinFile_(DTYPE* data, std::string target)
{

    std::string root = "./Weights/";
    std::string test = root+target + ".bin";
	std::cout<< test << std::endl;
//    if( data==NULL)
//    {
//		printf("Data is NULL\n");
//		exit(1);
//	}
//
//    int index=0;
//    DTYPE load_val;
//    std::ifstream read_file(test, std::ios::binary);
//    if ( !read_file.is_open() )
//    {
//        std::cout<<"No Such Binaray"<<std::endl;
//        exit(-1);
//    }
//    while( read_file.read(reinterpret_cast<char*>(&load_val), sizeof(DTYPE)))
//    {
//        data[index++] = load_val;
//    }
}

