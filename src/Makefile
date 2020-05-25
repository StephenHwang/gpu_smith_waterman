C = nvcc
NVCCFLAGS = -arch=sm_60 
CFLAGS = -std=c++11

all: align

align: align.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o align.exe align.cu 

clean:
	rm -f align.exe *.dat

