#!/bin/bash

mpicc lab1/$1.c -o $1
mpiexec -n $2 $1 $3 $4 $5