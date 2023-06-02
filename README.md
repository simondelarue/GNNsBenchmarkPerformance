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
| `Cora` | 2,708 | 10,556 | 2.880e-03 | 1,433 | 49,216 | 1.268e-02 | 7 |  

| Dataset      |
|--------------|
| [```SF2H```](http://www.sociopatterns.org/datasets/sfhh-conference-data-set/) |
| [```HighSchool```](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) |  
| [```ia-contact```](https://networkrepository.com/ia-contact.php) | 
| [```ia-contacts-hypertext2009```](http://www.sociopatterns.org/datasets/hypertext-2009-dynamic-contact-network/) |
| [```ia-enron-employees```](https://networkrepository.com/ia_enron_employees.php) |
