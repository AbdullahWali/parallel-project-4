#!/bin/bash
k_vals=(1000000 5000000 10000000)
num_threads=(32 64 128 256 512)
nvcc dot_product.cu -o dot_product
for i in ${k_vals[@]}; 
do
	for j in ${num_threads[@]};
	do
		for k in `seq 1 4`
		do
			echo "k = ${i} with number of threads = ${j}. Output is written into results.txt, repeat: ${k}."
			echo "k = ${i} with number of threads = ${j}. Output is written into results.txt, repeat: ${k}." >> results.txt
			./dot_product ${i}  ${j} >> results.txt
		done
	done
done
