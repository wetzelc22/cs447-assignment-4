#!/bin/bash
#SBATCH --job-name=numeric_integration
#SBATCH --cpus-per-task=24
#SBATCH -N 1
#SBATCH --mem=8G

A=0
B=1
iterations=100000000
n_threads=36
trials=4

for((j=1;j<=$trials;j++));
do
    for((i=1;i<=$n_threads;i++));
    do
	(/home/kchiu/time -p ./integrate $A $B $iterations $i) &>> output-$j.txt
    done
done
