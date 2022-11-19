PROGRAM=ex5a

TARGETS=$(PROGRAM)

CC = gcc
# CC = clang

CFLAGS = -fopenmp

LIBS  = -lmkl_sequential -lmkl_core -lmkl_rt -lpthread -lm 
EXTRA = -std=c99 -O3 -Wall -Wno-unused 
INCS  = 

$(PROGRAM): $(PROGRAM).c
	$(CC) $(CFLAGS) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

