#include <iostream>
#include <fstream>
#include <math.h>
#include <assert.h>
#include "network.h" 
#include "utils.h"
#include "convolution.cuh"
#include "Debug.h"

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
	std::cout << "read done" <<std::endl;
}


void Test()
{
	Network network;
	GetMobileNetV2(network);
	//network.PrintNetwork();
	//network.Forward();
	Matrix input(1,3,224,224);
	ReadBinFile("./Weights/input/input_data.bin",input.data);

	Matrix output = network.Forward(input, 0,0);
}

int main()
{
	Test();
	return 0;
}


