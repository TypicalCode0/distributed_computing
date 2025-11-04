#!/bin/bash

mpicc lab1/$1.c -o $1
mpiexec mpiexec -x PMIX_MCA_gds=hash -n $2 $1 $3 $4 $5
