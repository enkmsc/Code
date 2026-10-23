#!/bin/bash

for d in ./POSCAR* ; do
echo "$d"
cd $d
mlp3 convert --input_format=outcar --append OUTCAR ../br.cfg
cd ..
done
