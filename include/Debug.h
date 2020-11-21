#pragma once
#define DEBUG 1

#define print(fmt, args...)\
	do{\
		if(DEBUG)\
			fprintf(stdout,fmt,##args);\
	}while(0)

