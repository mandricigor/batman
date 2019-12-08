
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
from networkx.algorithms import bipartite


def parse_args():
    main_p = argparse.ArgumentParser()
    main_p.add_argument('-data', dest='datasets', required=True, help='comma separated list of paths to the datasets to be integrated')
    main_p.add_argument('-output', dest='outputdata', required=True, help='path for the output file')
    main_p.add_argument('-anchors', dest='ancnum', default=20, required=False, help='number of anchor points to consider in each dataset')
    main_p.add_argument('-clusters', dest='clusters', default=30, required=False, help='number of anchor points to consider in each dataset')
    main_p.add_argument('-neighbors', dest='neighbors', default=3, required=False, help='number of nearest anchors to consider')
    main_p.add_argument('-smoothing', dest='smoothing', default=0.5, required=False, help='smoothing coefficient')
    main_p.add_argument('-filter-cor', dest='filter_cor', default=0.7, required=False, help='filtering threshold on Pearson correlation between the anchors across the datasets')
    main_p.add_argument('-greedy', dest='greedy', required=False, action='store_true', help='use greedy heuristic for the minimum weight matching (recommended for extra large datasets)')
    main_p.add_argument('-rowgenes', dest='rowgenes', required=False, action='store_true', help='rows of the data matrix are genes')
    return vars(main_p.parse_args()) 


args = parse_args()

filter_cor = float(args["filter_cor"])
ancnum = int(args["ancnum"])
neighbors = int(args["neighbors"])
smoothing = float(args["smoothing"])
greedy = args["greedy"]
rowgenes = args["rowgenes"]
outputdata = args["outputdata"]
datasets = args["datasets"]
clusters = int(args["clusters"])
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

if rowgenes:
    data1 = data1.T
    data2 = data2.T

logger.info("Running KMeans on dataset 1")
km1 = MiniBatchKMeans(n_clusters=clusters, random_state=0).fit(data1)
data11 = km1.cluster_centers_

logger.info("Running KMeans on dataset 2")
km2 = MiniBatchKMeans(n_clusters=clusters, random_state=0).fit(data2)
data22 = km2.cluster_centers_

cluster_freq1 = Counter(km1.labels_)
cluster_freq2 = Counter(km2.labels_)

A_labels = {}
B_labels = {}

for x, y in enumerate(km1.labels_):
    if y not in A_labels:
        A_labels[y] = []
    A_labels[y].append(x)

for x, y in enumerate(km2.labels_):
    if y not in B_labels:
        B_labels[y] = []
    B_labels[y].append(x)

cluster_threshold = 0
cluster_bad1 = []
for x, y in cluster_freq1.items():
    if y < cluster_threshold:
        cluster_bad1.append(x)

cluster_bad2 = []
for x, y in cluster_freq2.items():
    if y < cluster_threshold:
        cluster_bad2.append(x)

for x in cluster_bad1:
    del cluster_freq1[x]
for x in cluster_bad2:
    del cluster_freq2[x]
    
s1 = sum(cluster_freq1.values())
s2 = sum(cluster_freq2.values())

cluster_share1 = {}
for x, y in cluster_freq1.items():
    cluster_share1[x] = int(round(1.0 * ancnum * y / s1))

cluster_share2 = {}
for x, y in cluster_freq2.items():
    cluster_share2[x] = int(round(1.0 * ancnum * y / s2))

graph = nx.Graph()
nodes1 = []
for x in cluster_share1.keys():
    for i in range(cluster_share1[x]):
        nodes1.append("%s_A_%s" % (x, i))

nodes2 = []
for x in cluster_share2.keys():
    for i in range(cluster_share2[x]):
        nodes2.append("%s_B_%s" % (x, i))


graph = nx.Graph()
graph.add_nodes_from(nodes1, bipartite=0)
graph.add_nodes_from(nodes2, bipartite=1)

for x in nodes1:
    for y in nodes2:
        xx = km1.cluster_centers_[int(x.split("_")[0]), :]
        yy = km2.cluster_centers_[int(y.split("_")[0]), :]
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

graph2 = nx.Graph()
nodes11 = []
nodes22 = []
for x in cluster_share1.keys():
    nodes11.append("%s_A" % x)
for x in cluster_share2.keys():
    nodes22.append("%s_B" % x)

graph2.add_nodes_from(nodes11, bipartite=0)
graph2.add_nodes_from(nodes22, bipartite=1)

for x, y in mat:
    if "B" in x:
        x, y = y, x
    x, y = "_".join(x.split("_")[:2]), "_".join(y.split("_")[:2])
    if (x, y) not in graph2.edges():
        graph2.add_edge(x, y)


# filter the graph
B_fromA = {}
for node in nodes22:
    orignode = int(node.split("_")[0])
    for point in B_labels[orignode]:
        center_dist = {}
        for center in nx.neighbors(graph2, node):
            orig_center = int(center.split("_")[0])
            center_dist[center] = sp.spatial.distance.euclidean(data2[point, :], km1.cluster_centers_[orig_center, :])
        if center_dist:
            mindist_center = min(center_dist.items(), key = lambda x: x[1])
            if node not in B_fromA:
                B_fromA[node] = set()
            B_fromA[node].add(mindist_center[0])

A_fromB = {}
for node in nodes11:
    orignode = int(node.split("_")[0])
    for point in A_labels[orignode]:
        center_dist = {}
        for center in nx.neighbors(graph2, node):
            orig_center = int(center.split("_")[0])
            center_dist[center] = sp.spatial.distance.euclidean(data1[point, :], km2.cluster_centers_[orig_center, :])
        if center_dist:
            mindist_center = min(center_dist.items(), key = lambda x: x[1])
            if node not in A_fromB:
                A_fromB[node] = set()
            A_fromB[node].add(mindist_center[0])

edges_to_keep = set()

for x in A_fromB:
    for y in A_fromB[x]:
        if x in B_fromA[y]:
            edges_to_keep.add((x, y))

for x in B_fromA:
    for y in B_fromA[x]:
        if x in A_fromB[y]:
            edges_to_keep.add((y, x))


edges_to_delete = set()
for x, y in graph2.edges():
    if (x, y) in edges_to_keep or (y, x) in edges_to_keep:
        pass
    else:
        edges_to_delete.add((x, y))


for x, y in edges_to_delete:
    graph2.remove_edge(x, y)

anchors = []
emgraphs = []
conncomp = nx.connected_components(graph2)
for cc in list(conncomp):
    if len(cc) == 1:
        pass
    elif len(cc) == 2:
        # we have an anchor pair!
        x, y = cc
        if "B" in x:
            x, y = y, x
        x = int(x.split("_")[0])
        y = int(y.split("_")[0])
        anchors.append((km1.cluster_centers_[x, :], km2.cluster_centers_[y, :]))
    else:
        emgraphs.append(nx.subgraph(graph2, cc))

for emg in emgraphs:
    A_anch = {}
    B_anch = {}
    for node in emg.nodes():
        if nx.degree(emg, node) > 1:
            #this is an interesting node
            orignode = int(node.split("_")[0])
            points_of_centers = {}
            if "A" in node:
                for point in A_labels[orignode]:
                    center_dist = {}
                    for center in nx.neighbors(emg, node):
                        orig_center = int(center.split("_")[0])
                        center_dist[center] = sp.spatial.distance.euclidean(data1[point, :], km2.cluster_centers_[orig_center, :])
                    mindist_center = min(center_dist.items(), key = lambda x: x[1])       
                    if mindist_center[0] not in points_of_centers:
                        points_of_centers[mindist_center[0]] = set()
                    points_of_centers[mindist_center[0]].add(point)
                if node not in A_anch:
                    A_anch[node] = {}
                for x, y in points_of_centers.items():
                    A_anch[node][x] = np.mean(data1[list(y), :], axis = 0)
            elif "B" in node:
                for point in B_labels[orignode]:
                    center_dist = {}
                    for center in nx.neighbors(emg, node):
                        orig_center = int(center.split("_")[0])
                        center_dist[center] = sp.spatial.distance.euclidean(data2[point, :], km1.cluster_centers_[orig_center, :])
                    mindist_center = min(center_dist.items(), key = lambda x: x[1]) 
                    if mindist_center[0] not in points_of_centers:
                        points_of_centers[mindist_center[0]] = set()
                    points_of_centers[mindist_center[0]].add(point)
                if node not in B_anch:
                    B_anch[node] = {}
                for x, y in points_of_centers.items():
                    B_anch[node][x] = np.mean(data2[list(y), :], axis = 0)
    for u, v in emg.edges():
        if "B" in u: 
            u, v = v, u
        if u in A_anch and v in B_anch:
            anchors.append((A_anch[u][v], B_anch[v][u]))
        elif u in A_anch and v not in B_anch:
            ornodeb = int(v.split("_")[0])
            anchors.append((A_anch[u][v], km2.cluster_centers_[ornodeb, :]))
        elif u not in A_anch and v in B_anch:
            ornodea = int(u.split("_")[0])
            anchors.append((km1.cluster_centers_[ornodea, :], B_anch[v][u]))

intvectors = []
logger.info("integrating dataset 1 into 2")
for i in range(data1.shape[0]):
    yy = []
    for iii, xx in enumerate(anchors):
        yy.append((iii, sp.spatial.distance.euclidean(data1[i, :], xx[0])))
    yy = sorted(yy, key = lambda z: z[1])
    clnei = neighbors
    closest = []
    for ii in range(clnei):
        closest.append(yy[ii][0])
    dist = []
    for ii in range(clnei):
        dist.append(yy[ii][1])
    u = smoothing
    sumportions = 0
    for d in dist:
        sumportions += math.exp(-u * d)
    portion = []
    for d in dist:
        portion.append(math.exp(-u * d) / sumportions)
    intvect = np.zeros(len(anchors[0][0]))
    for v, w in zip(portion, closest):
        intvect += v * (anchors[w][1] - anchors[w][0])
        #intvect += 2 * (anchors[w][1])
    intvectors.append(intvect)

for i in range(data1.shape[0]):
    data1[i, :] += intvectors[i]

logger.info("saving the results")
np.savetxt(outputdata, np.vstack([data1, data2]), fmt = "%.12f")

logger.info("your datasets have been integrated!")
logger.info("thank you!")
