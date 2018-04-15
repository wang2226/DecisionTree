#!/bin/bash
for i in 1 4 7 10 13 16 19 22 25 28 31 34
do
	echo $i
	echo $(python decisiontree.py ./adult.data ./adult.test depth 20 40 $i) 
done

