
import sys
import time
import math
import argparse
import logging
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import networkx as nx
import time
from collections import Counter
import WBbM as WBbM

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


data1 = data1.T
data2 = data2.T

logger.info("Finding the anchors points in dataset 1")
km1 = MiniBatchKMeans(n_clusters=ancnum, random_state=0).fit(data1)
data11 = km1.cluster_centers_

logger.info("Finding the anchors points in dataset 2")
km2 = MiniBatchKMeans(n_clusters=ancnum, random_state=0).fit(data2)
data22 = km2.cluster_centers_


data11 = km1.cluster_centers_
data22 = km2.cluster_centers_


la = len(data11)
lb = len(data22)


logger.info("building the anchor graph")
graph = nx.Graph()
nodes1 = list(range(la))
nodes2 = np.array(range(la, la + lb))
graph.add_nodes_from(nodes1, bipartite=0)
graph.add_nodes_from(nodes2, bipartite=1)
distances = sp.spatial.distance.cdist(data11, data22, 'euclidean')


for i in range(len(nodes1)):
    xx = []
    for j in range(len(nodes2)):
        cor = pearsonr(data11[i, :], data22[j, :])[0]
        xx.append((j, cor))
    xx = sorted(xx, key = lambda x: x[1], reverse = True)
    top10percent = int(len(xx) / 5)
    for j, _ in [(u, v) for u, v in xx[:top10percent] if v > filter_cor]:
        graph.add_edge(nodes1[i], nodes2[j], weight=-distances[i][j])


isola = list(nx.isolates(graph))[:]
for iso in isola:
    graph.remove_node(iso)


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
    if x >= la:
        x, y = y, x
    matching.append((x, y - la))
matching = sorted(matching, key = lambda x: x[0])

logger.info("computing integration vectors")
integration_vectors = {}
for x, y in matching:
    integration_vectors[x] = data22[y, :] - data11[x, :]

logger.info("integration of dataset 1 into 2")
for i in range(data1.shape[0]):
    yy = []
    for x in integration_vectors.keys():
        yy.append((x, sp.spatial.distance.euclidean(data1[i, :], data11[x, :])))
    yy = sorted(yy, key = lambda z: z[1])
    closest1, closest2 = yy[0][0], yy[1][0]
    dist1, dist2 = yy[0][1], yy[1][0]
    portion1 = math.exp(-dist1) / (math.exp(-dist1) + math.exp(-dist2))    
    portion2 = math.exp(-dist2) / (math.exp(-dist1) + math.exp(-dist2))
    data1[i, :] += (portion1 * integration_vectors[closest1] + portion2 * integration_vectors[closest2])

logger.info("saving the results")
np.savetxt(outputdata, np.vstack([data1, data2]), fmt = "%.2f")


logger.info("thank you!")
