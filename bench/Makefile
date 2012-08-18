# GNU
CC=gcc-mp-4.7
#CFLAGS=-ftree-vectorizer-verbose=7 -std=c99 -O3
CFLAGS=-O3 -lgfortran -g -Wall -march=native -fPIC -DCYTHON_VECTOR_SIZE=4 -ftree-vectorize -msse4.2 -fopenmp

FC=gfortran-mp-4.7
FFLAGS=-O3 -fno-underscoring -fno-second-underscore -g -Wall -march=native -fPIC -fopenmp

CYTHON=`which cython`
# INTEL
# CC=icc
# CFLAGS=-O3 -openmp
# 
# FC=ifort
# FFLAGS=-O3 -nofor_main -openmp
# 
# LD=ifort #cc
# #LDFLAGS=-L/exports/applications/apps/SL5/intel/Compiler/11.0/081/lib/intel64/ -lifcore -limf
# LDFLAGS=-L/opt/intel/lib -L/exports/applications/apps/intel/ClusterStudio2011/lib/intel64 -lifcore -limf -liomp5

# NAGFOR (http://www.nag.com/nagware/np/r53_doc/nagfor.html)
# FC=nagfor
# FFLAGS=-O4 -pic
# LD=nagfor -Bstatic

PY=python2.7
BUILD_PY=python2.7

all: cy
	OMP_NUM_THREADS=4 $(PY) -c 'import bench; bench.run()'

new:
	$(PY) $(CYTHON) bench.pyx
	make

cy: fbench
	$(PY) $(CYTHON) bench.pyx
	CC=$(CC) LD="$(LD)" LDFLAGS="$(LDFLAGS)" CFLAGS="$(CFLAGS)" $(BUILD_PY) setup.py build_ext --inplace

fbench:
	$(FC) $(FFLAGS) -c fbench.f90

clean:
	rm -rf fbench.o bench.c bench.so build *.o* *.e* *.pe* *.po*
