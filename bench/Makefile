# GNU
CC=gcc-mp-4.7
#CFLAGS=-ftree-vectorizer-verbose=7 -std=c99 -O3
CFLAGS=-O3 -lgfortran -g -Wall -mcpu=native

FC=gfortran-mp-4.7
FFLAGS=-O3 -fno-underscoring -fno-second-underscore -g -Wall -march=native

# INTEL
CC=icc
CFLAGS=-O3

FC=ifort
FFLAGS=-O3 -nofor_main

LD=ifort #cc
LDFLAGS=-L/exports/applications/apps/SL5/intel/Compiler/11.0/081/lib/intel64/ -lifcore -limf

# NAGFOR (http://www.nag.com/nagware/np/r53_doc/nagfor.html)
# FC=nagfor
# FFLAGS=-O4 -pic
# LD=nagfor -Bstatic

PY=python
BUILD_PY=python

all: cy
	$(PY) -c 'import bench; bench.run()'

new:
	$(PY) `which cython` bench.pyx
	make

cy: fbench
	$(PY) `which cython` bench.pyx
	CC=$(CC) LD="$(LD)" LDFLAGS="$(LDFLAGS)" CFLAGS="$(CFLAGS)" $(BUILD_PY) setup.py build_ext --inplace
	# Distutils in python2.4 seems to ignore the LD environment variable...
	# icc -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -O3 -fPIC -I/exports/home/s1149948/apps/lib/python2.7/site-packages/numpy/core/include -I/exports/home/s1149948/apps/include/python2.7 -c bench.c -o build/temp.linux-x86_64-2.7/bench.o

fbench:
	$(FC) $(FFLAGS) -c fbench.f90

clean:
	rm -rf fbench.o bench.c bench.so build *.o* *.e* *.pe* *.po*
