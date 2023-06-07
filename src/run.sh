#!/bin/sh

for i in {5..5}
do
    # Baselines
    # ---------
    # Using Structure
    # ---------------
    # Undirected + With penalization
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true

    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true

    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN

    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN

    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true

    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true

    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true

    # Using features instead of structure
    # -----------------------------------
    # Undirected + With penalization
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank --use_features=true
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation --use_features=true
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion --use_features=true
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --use_features=true
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true --use_features=true

    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank --use_features=true
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation --use_features=true
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion --use_features=true
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --use_features=true
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true --use_features=true

    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank --use_features=true
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation --use_features=true
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion --use_features=true
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true --use_features=true

    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank --use_features=true
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation --use_features=true
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion --use_features=true
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --use_features=true
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true --use_features=true

    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank --use_features=true
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation --use_features=true
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion --use_features=true
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --use_features=true
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true --use_features=true

    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank --use_features=true
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation --use_features=true
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion --use_features=true
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --use_features=true
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true --use_features=true

    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank --use_features=true
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation --use_features=true
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion --use_features=true
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --use_features=true
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN --embedding_method=true --use_features=true
    
    # Directed + with penalization
    # --> directed graph doest not exist in Sknetwork
    #python main.py --dataset=citeseer --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=citeseer --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=citeseer --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=citeseer --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN

    #python main.py --dataset=pubmed --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=PageRank
    #python main.py --dataset=pubmed --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=LabelPropagation
    #python main.py --dataset=pubmed --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=Diffusion
    #python main.py --dataset=pubmed --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=KNN

    # GNNs
    # ----
    # Undirected + With penalization
    python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=pubmed --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=reddit --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=wikivitals --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=wikivitals-fr --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=wikischools --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    # Directed + with penalization
    # --> directed graph doest not exist in Sknetwork
    #python main.py --dataset=citeseer --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=citeseer --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=citeseer --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=citeseer --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    #python main.py --dataset=pubmed --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN
    #python main.py --dataset=pubmed --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GraphSage
    #python main.py --dataset=pubmed --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=GAT
    #python main.py --dataset=pubmed --undirected=false --penalized=true --randomstate=8 --k=$i --stratified=true --model=SGC

    # Scikit Network GNNs
    # ----
    # Undirected + With penalization
    #python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN_skn
    #python main.py --dataset=citeseer --undirected=true --penalized=true --randomstate=8 --k=$i --stratified=true --model=GCN_skn
done