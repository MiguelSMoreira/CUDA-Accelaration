CC		:= g++
PROJ_BASE	:= ..
EXEC		:= $(PROJ_BASE)/bin/sequential
IMAGE		:=$(PROJ_BASE)/images/image15.jpg
INCLUDES	:= -I"$(PROJ_BASE)/include"
LDFLAGS         := -lm -lX11 -lpthread

debug		:= 0
ifneq ($(debug), 1)
	CFLAGS		:= -O3 -g0 
else
	CFLAGS		:= -O0 -g3 -Wall
endif

all: clean sequential run

sequential: main.cpp filters.cpp
	$(CC) -o $(EXEC) main.cpp filters.cpp ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

run: 
	$(EXEC) $(IMAGE)

clean:
	rm -f $(EXEC) *~


