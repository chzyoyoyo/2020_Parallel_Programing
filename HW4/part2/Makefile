main: main.o matmul.o
	mpicxx -o matmul main.o matmul.o
main.o: main.cc
	mpicxx -c main.cc
matmul.o: matmul.cc
	mpicxx -c matmul.cc
