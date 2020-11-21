CC=gcc
CXX=g++
CPPFLAGS= -Wall -g -c  -fPIC -std=c++14 -I./include

OBJS =   ./obj/*.o

#TARGET = ./mobilenetv2

./mobilenetv2 : ./obj/network.o ./obj/utils.o ./obj/mobilenetv2.o  
	g++ -Wall -g -std=c++14 -I./include -o ./mobilenetv2 $(OBJS)

./obj/utils.o : ./src/utils.cpp
	$(CXX) $(CPPFLAGS) -o ./obj/utils.o ./src/utils.cpp

./obj/network.o : ./src/network.cpp
	$(CXX) $(CPPFLAGS) -o ./obj/network.o ./src/network.cpp

./obj/mobilenetv2.o : ./src/mobilenetv2.cpp
	$(CXX) $(CPPFLAGS) -o ./obj/mobilenetv2.o ./src/mobilenetv2.cpp 


clean :
	rm -rf $(OBJS)


