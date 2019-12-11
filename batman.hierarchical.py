import sys
import time
import math
import argparse
import logging
import numpy as np
import scipy as sp
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import pearsonr
import networkx as nx
import time
from collections import Counter
#from equal_groups import EqualGroupsKMeans


def parse_args():
    main_p = argparse.ArgumentParser()
    main_p.add_argument('-data', dest='datasets', required=True, help='comma separated list of paths to the datasets to be integrated')
    main_p.add_argument('-output', dest='outputdata', required=True, help='path for the output file')
    main_p.add_argument('-anchors', dest='ancnum', default=200, required=False, help='number of anchor points to consider in each dataset')
    main_p.add_argument('-filter-cor', dest='filter_cor', default=0.7, required=False, help='filtering threshold on Pearson correlation between the anchors across the datasets')
    main_p.add_argument('-greedy', dest='greedy', required=False, action='store_true', help='use greedy heuristic for the minimum weight matching (recommended for extra large datasets)')
    return vars(main_p.parse_args()) 


args = parse_args()

filter_cor = float(args["filter_cor"])
ancnum = int(args["ancnum"])
greedy = args["greedy"]
outputdata = args["outputdata"]
datasets = args["datasets"]
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

datasets = datasets.split(",")
logger.info("loading the datasets into the memory")
data1 = np.loadtxt(datasets[0])
data2 = np.loadtxt(datasets[1])




km1 = MiniBatchKMeans(n_clusters=5, random_state=0).fit(data1)
km1.labels_ = np.array(map(str, km1.labels_), dtype=object)
km1_cluster_centers_ = {}
for i in range(km1.cluster_centers_.shape[0]):
    km1_cluster_centers_[str(i)] = km1.cluster_centers_[i, :]
for i in range(20):
    cluster_freq1 = Counter(km1.labels_)
    bigs = [x for x, y in cluster_freq1.items() if y > 56]
    print bigs, "BIGS"
    for u, big in enumerate(bigs):
        print "BIG", big, type(big)
        km11 = MiniBatchKMeans(n_clusters=5, random_state=0).fit(data1[km1.labels_ == big])
        for kl in set(km11.labels_):
            km1_cluster_centers_["%s:%s" % (big, kl)] = np.mean(data1[km1.labels_ == big, :][km11.labels_ == kl, :], axis = 0)
        del km1_cluster_centers_[big]
        km1.labels_[km1.labels_ == big] = list(map(lambda xxx: "%s:%s" % (big, xxx), (km11.labels_)))

km2 = MiniBatchKMeans(n_clusters=5, random_state=0).fit(data2)
km2.labels_ = np.array(map(str, km2.labels_), dtype=object)
km2_cluster_centers_ = {}
for i in range(km2.cluster_centers_.shape[0]):
    km2_cluster_centers_[str(i)] = km2.cluster_centers_[i, :]
for i in range(20):
    cluster_freq2 = Counter(km2.labels_)
    bigs = [x for x, y in cluster_freq2.items() if y > 40]
    print bigs, "BIGS"
    for u, big in enumerate(bigs):
        print "BIG", big, type(big)
        km22 = MiniBatchKMeans(n_clusters=5, random_state=0).fit(data2[km2.labels_ == big])
        for kl in set(km22.labels_):
            km2_cluster_centers_["%s:%s" % (big, kl)] = np.mean(data2[km2.labels_ == big, :][km22.labels_ == kl, :], axis = 0)
        del km2_cluster_centers_[big]
        km2.labels_[km2.labels_ == big] = list(map(lambda xxx: "%s:%s" % (big, xxx), (km22.labels_)))


cluster_freq1 = Counter(km1.labels_)
cluster_freq2 = Counter(km2.labels_)

del1 = []
for x, y in cluster_freq1.items():
    if y < 3:
        del1.append(x)
del2 = []
for x, y in cluster_freq2.items():
    if y < 3:
        del2.append(x)

for x in del1:
    del cluster_freq1[x]
for x in del2:
    del cluster_freq2[x]

graph = nx.Graph()
nodes1 = []
for x in cluster_freq1.keys():
    for i in range(cluster_freq1[x]):
        nodes1.append(x + "_A")

nodes2 = []
for x in cluster_freq2.keys():
    for i in range(cluster_freq2[x]):
        nodes2.append(x + "_B")


graph = nx.Graph()
graph.add_nodes_from(nodes1, bipartite=0)
graph.add_nodes_from(nodes2, bipartite=1)


for x in nodes1:
    for y in nodes2:
        xx = km1_cluster_centers_[x.split("_")[0]]
        yy = km2_cluster_centers_[y.split("_")[0]]
        xydist = sp.spatial.distance.euclidean(xx, yy)
        graph.add_edge(x, y, weight=-xydist)




logger.info("matching anchors via minimum weight matching")
if greedy:
    def greedy_matching(graph, nr_edges=0):
        from queue import PriorityQueue
        pq = PriorityQueue()
        for x, y in graph.edges():
            pq.put((-graph[x][y]["weight"], (x, y)))
        matchings = {}
        while not pq.empty() or len(matchings) < nr_edges:
            el = pq.get()
            w, el = el
            if not matchings.get(el[0]) and not matchings.get(el[1]):
                matchings[el[0]] = el[1]
                matchings[el[1]] = el[0]
        matchings = set(map(lambda x: tuple(sorted(x)), matchings.items()))
        return matchings
    mat = greedy_matching(graph)
else:
    mat = nx.max_weight_matching(graph, maxcardinality=True)


matching = []
for x, y in mat:
    if "B" in x:
        x, y = y, x
    matching.append((x, y))


logger.info("computing integration vectors")
anchors = []
for x, y in matching:
    x = x.split("_")[0]
    y = y.split("_")[0]
    anchors.append((km1_cluster_centers_[x], km2_cluster_centers_[y]))





intvectors = []
logger.info("integrating dataset 1 into 2")
for i in range(data1.shape[0]):
    yy = []
    for iii, xx in enumerate(anchors):
        yy.append((iii, sp.spatial.distance.euclidean(data1[i, :], xx[0])))
    yy = sorted(yy, key = lambda z: z[1])
    clnei = 10
    closest = []
    for ii in range(clnei):
        closest.append(yy[ii][0])
    dist = []
    for ii in range(clnei):
        dist.append(yy[ii][1])
    u = 0.5
    sumportions = 0
    for d in dist:
        sumportions += math.exp(-u * d)
    portion = []
    for d in dist:
        portion.append(math.exp(-u * d) / sumportions)
    intvect = np.zeros(len(anchors[0][0]))
    for v, w in zip(portion, closest):
        intvect += v * (anchors[w][1] - anchors[w][0])
    intvectors.append(intvect)
    data1[i, :] += intvect



with open("centers1.csv", "w") as f:
    for x, y in km1_cluster_centers_.items():
        f.write("%s %s\n" % (y[0], y[1]))

with open("centers2.csv", "w") as f:
    for x, y in km2_cluster_centers_.items():
        f.write("%s %s\n" % (y[0], y[1]))



with open("mat1.csv", "w") as f:
    for x, y in anchors:
        f.write("%s %s\n" % (x[0], x[1]))

with open("mat2.csv", "w") as f:
    for x, y in anchors:
        f.write("%s %s\n" % (y[0], y[1]))

with open("vectors.csv", "w") as f:
    for i in range(data1.shape[0]):
        f.write("%s %s\n" % (intvectors[i][0], intvectors[i][1]))

logger.info("saving the results")
np.savetxt(outputdata, np.vstack([data1, data2]), fmt = "%.12f")


logger.info("thank you!")
