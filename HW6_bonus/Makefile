default: conv

CC = gcc-10
NVCC = nvcc
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g

OBJS = main.o bmpfuncs.o serialConv.o helper.o hostFE.o

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

conv: $(OBJS)
	$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS)

hostFE.o : hostFE.cu hostFE.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c hostFE.cu -o $@

bmpfuncs.o: bmpfuncs.c
	$(CC) -c $(FLAGS) $< -o $@
serialConv.o: serialConv.c
	$(CC) -c $(FLAGS) $< -o $@
helper.o: helper.c
	$(CC) -c $(FLAGS) $< -o $@


clean:
	rm -f conv *.o output.bmp ref.bmp