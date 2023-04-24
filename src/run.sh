#!/bin/sh

for i in {3..15}
do
    # Baselines
    python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN
done