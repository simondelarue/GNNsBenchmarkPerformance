# Graph Neural Network benchmark with simple baselines

We compare famous Graph Neural Network based models with simple baselines. The motivation of this work is to provide a reproducible framework to evaluate recent advances in the GNN field. In recent literature, GNN are rarely compared to simple approaches; we argue that in some cases these baseline methods can achieve competitive results and would like to explore when exactly it is useful to leverage complex models.

## Usage
```shell
./run.sh
```

## Datasets

We compare model performances on commonly used `Planetoid` datasets, *i.e.* `Cora`, `Pubmed` and `Citeseer`. But most importantly, we report performance comparisons on larger Wikipedia-based real-world datasets, *i.e.* `Wikivitals`, `Wikivitals-fr` and `Wikischools` (available [here](https://netset.telecom-paris.fr/)).

Dataset statistics for undirected versions of the graphs:

| Name | #nodes | #edges adjacency | dA | #features | #edges biadjacency | dB | #classes |  
|:-----|--------|------------------|----|-----------|--------------------|----|----------|  
| ```Cora``` | 2,708 | 10,556 | 2.880e-03 | 1,433 | 49,216 | 1.268e-02 | 7 |  



| `CiteSeer` | 3,327 | 9,104 | 1.645e-03 | 3,703 | 105,165 | 8.536e-03 | 6 |  
| `PubMed` | 19,717 | 88,648 | 4.561e-04 | 500 | 988,031 | 1.002e-01 | 3 |  
| `Wikivitals` | 10,011 | 824,999 | 1.647e-02 | 37,845 | 1,363,301 | 3.598e-03 | 11 |  
| `Wikivitals-fr` | 9945 | 558,427 | 1.129e-02 | 28,198 | 873,555 | 3.115e-03 | 11 |  
| `Wikischools` | 4403 | 112,834 | 1.164e-02 | 20,527 | 474,138 | 5.246e-03 | 16 |  


| Dataset      |
|--------------|
| [```SF2H```](http://www.sociopatterns.org/datasets/sfhh-conference-data-set/) |
| [```HighSchool```](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) |  
| [```ia-contact```](https://networkrepository.com/ia-contact.php) | 
| [```ia-contacts-hypertext2009```](http://www.sociopatterns.org/datasets/hypertext-2009-dynamic-contact-network/) |
| [```ia-enron-employees```](https://networkrepository.com/ia_enron_employees.php) |
