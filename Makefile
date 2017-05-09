INC = -I$(CUDA_SDK_PATH)/common/inc
LIB = -L$(CUDA_SDK_PATH)/lib -lcutil
PG = -Xcompiler -DPROFILE_PG
PGPU = -Xcompiler -DPROFILE_GPU

all: hmm fhmm

cuhmm: hmm.cu
	nvcc $(INC) $(LIB) hmm.cu -o cuhmm

hmm: hmm.cpp
	g++ -std=c++11 -Wall -fopenmp -g -lm hmm.cpp -O2 -o hmm

fhmm: fhmm.c
	gcc -Wall -lm fhmm.c -o fhmm

clean: 
	rm -f hmm fhmm
