CC?=gcc

all: c2c

c2c: c2c.c
	$(CC) -O2 -fopenmp $< -o c2c

clean:
	rm -f c2c perf_*
