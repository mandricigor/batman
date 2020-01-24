# BATMAN
BATMAN: <ins>BAT</ins>ch effect correction via minimum weight <ins>MA</ins>tchi<ins>N</ins>g



BATMAN is a novel tool for batch effect correction (integration) of single-cell RNA-Seq datasets. Currently, it supports integration of two datasets. In order to get help on how to run BATMAN, please, type the following command in the terminal:

```bash
$ python3.7 batman.py --help
```

As a result, you will see the following:

```
usage: batman.py [-h] -data DATASETS -output OUTPUTDATA [-anchors ANCNUM]
                 [-filter-cor FILTER_COR] [-greedy]

optional arguments:
  -h, --help            show this help message and exit
  -data DATASETS        comma separated list of paths to the datasets to be
                        integrated
  -output OUTPUTDATA    path for the output file
  -anchors ANCNUM       number of anchor points to consider in each dataset
  -filter-cor FILTER_COR
                        filtering threshold on Pearson correlation between the
                        anchors across the datasets
  -greedy               use greedy heuristic for the minimum weight matching
                        (recommended for extra large datasets)
```


The main idea of BATMAN is based on finding a parsimonious matching between two sets of anchors in each of the two datasets. This is done by the following procedure:

1. Identifying the set of anchors in each of the datasets (their number is specified with parameter -anchors)
2. Building a weighted bipartite graph where each partition corresponds to each of the datasets and the edges are weighted by the Euclidean distance between the points (cells in N-dimensional space) corresponding to their endpoints
3. Finding minimum weight matching in this graph
    * a. Exact (blossom-based) algorithm
    * b. Greedy approximation (approximation ratio 2)
4. Determining local shifting vectors
5. Batch correction


BATMAN was developed by Igor Mandric at UCLA. If you have any questions or suggestions, feel free to contact the author at [imandric@ucla.edu](mailto:imandric@ucla.edu).


