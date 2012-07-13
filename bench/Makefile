CC=gcc-mp-4.7
#CFLAGS=-ftree-vectorizer-verbose=7 -std=c99 -O3
CFLAGS=-O3 -lgfortran -g -Wall -mcpu=native
#LDFLAGS=-lgomp

FC=gfortran-mp-4.7
FFLAGS=-O3 -fno-underscoring -fno-second-underscore -g -Wall -march=native

all: cy
	python -c 'import bench; bench.run()'

new:
	cython bench.pyx
	make

cy: fbench
	CC=$(CC) CFLAGS="$(CFLAGS)" python setup.py build_ext --inplace

fbench:
	$(FC) $(FFLAGS) -c fbench.f90

clean:
	rm -rf fbench.o bench.c bench.so build
