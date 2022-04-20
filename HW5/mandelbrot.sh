#!/bin/bash

echo "view 1"
echo "iter 1000"
for loop in 1 2 3 4 5 6 7 8 9 10; do
    ./mandelbrot -i 1000 -v 1 -g 1
done
echo "iter 10000"
for loop in 1 2 3 4 5 6 7 8 9 10; do
    ./mandelbrot -i 10000 -v 1 -g 1
done
echo "iter 100000"
for loop in 1 2 3 4 5 6 7 8 9 10; do
    ./mandelbrot -i 100000 -v 1 -g 1
done
echo "view 2"
echo "iter 1000"
for loop in 1 2 3 4 5 6 7 8 9 10; do
    ./mandelbrot -i 1000 -v 2 -g 1
done
echo "iter 10000"
for loop in 1 2 3 4 5 6 7 8 9 10; do
    ./mandelbrot -i 10000 -v 2 -g 1
done
echo "iter 100000"
for loop in 1 2 3 4 5 6 7 8 9 10; do
    ./mandelbrot -i 100000 -v  -g 1
done