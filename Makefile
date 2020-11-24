CC=gcc
CXX=g++
NVCC=nvcc
CUDA_INC=/usr/local/cuda/include
CUDA_LIBS=/usr/local/cuda/lib64

CPPFLAGS= -Wall -g -c  -fPIC -std=c++14 -I./include
NVCCFLAGS= -g -c -lcuda -lcudart -arch=sm_61

OBJS =   ./obj/*.o

#TARGET = ./mobilenetv2

./mobilenetv2 : ./obj/network.o ./obj/convolution.o ./obj/utils.o ./obj/mobilenetv2.o
	nvcc -o $@  $^ -I./include

./obj/utils.o : ./src/utils.cpp
	$(CXX) $(CPPFLAGS) -o ./obj/utils.o ./src/utils.cpp

./obj/mobilenetv2.o : ./src/mobilenetv2.cpp
	$(CXX) $(CPPFLAGS) -o ./obj/mobilenetv2.o ./src/mobilenetv2.cpp 

./obj/convolution.o : ./src/convolution.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -I$(CUDA_INC) -I./include -L$(CUDA_LIBS) 

./obj/network.o : ./src/network.cpp  
	$(NVCC) $(NVCCFLAGS) -o ./obj/network.o ./src/network.cpp  -I./include



clean :
	rm -rf $(OBJS)


