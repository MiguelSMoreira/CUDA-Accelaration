CC		:= g++
NVCC		:= nvcc
PROJ_BASE	:= ..
EXEC		:= $(PROJ_BASE)/bin/cuda
IMAGE		:=$(PROJ_BASE)/images/image15.jpg

INCLUDES	:= -I"$(PROJ_BASE)/include"

LDFLAGS         := -lm -lX11 -lpthread
CUDA_LDFLAGS	:= $(LDFLAGS) -lrt -lcudart

LIBS		:=
CUDA_LIBS	:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

################################################################################
# CUDA code generation flags
#GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_SM32    := -gencode arch=compute_32,code=sm_32
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
#GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) \
                $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50)

NVCCFLAGS	:= $(GENCODE_FLAGS)

debug		:= 0 
ifneq ($(debug), 1)
	CFLAGS		:= -O3 -g0 
	NVCCFLAGS	+= -O3
else
	CFLAGS		:= -O0 -g3 -Wall
	NVCCFLAGS	+= -O0
endif

all: $(EXEC)
 
$(EXEC): filters.o main.o Timer.o
	$(CC) -o $(EXEC) main.o filters.o Timer.o $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS) 

main.o: main.cpp
	$(CC) $(INCLUDES) $(CFLAGS) -c -o main.o main.cpp

Timer.o: ../src/Timer.cpp
	$(CC) $(INCLUDES) $(CFLAGS) -c -o Timer.o ../src/Timer.cpp

filters.o: filters.cu
	$(NVCC) -c filters.cu $(NVCCFLAGS) $(INCLUDES)

run: all 
	$(EXEC) $(IMAGE)

RUNSERVER_REQ_RUN_SETTINGS=True
RUNSERVER_DEPS=all
-include $(ACS_SHARED_PATH)/runserver.mk

clean:
	rm -f $(EXEC) *.o *~


