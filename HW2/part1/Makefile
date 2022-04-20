all:
	clang++-11 -I./common -O3 -std=c++17 -Wall -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -mavx2 pi.cpp -lpthread -o pi.out 