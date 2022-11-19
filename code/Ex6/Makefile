PROGRAM=cholesky

TARGETS=$(PROGRAM)

CC = gcc
# CC = clang

CFLAGS = -fopenmp
# -I$(MKL_INC_DIR)
# -L$(MKL_LIB_DIR) 
# METER NO LIBS depois -lm -lpthread
LIBS  = -lmkl_sequential -lmkl_core -lmkl_rt -lpthread -lm
EXTRA = -std=c99 -O3 -Wall -Wno-unused 
INCS  = 

$(PROGRAM): $(PROGRAM).c
	$(CC) $(CFLAGS) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 

