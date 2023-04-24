#!/bin/sh

for i in {3..15}
do
    python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
done