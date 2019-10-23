
import time
import math
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import networkx as nx
import time
from collections import Counter
import WBbM as WBbM

#data1 = np.loadtxt("ImmVar.csv")
#data2 = np.loadtxt("CLUES.csv")

data1 = np.loadtxt("smartseq2.csv")
data2 = np.loadtxt("celseq2.csv")


data1 = data1.T
data2 = data2.T

print data1.shape
print data2.shape

km1 = KMeans(n_clusters=200, random_state=0).fit(data1)
data11 = km1.cluster_centers_

km2 = KMeans(n_clusters=200, random_state=0).fit(data2)
data22 = km2.cluster_centers_



aa = sorted(Counter(km1.labels_).values(), reverse=True)
bb = sorted(Counter(km2.labels_).values(), reverse=True)

aa2 = [x * 1.0 / sum(aa) for x in aa]
bb2 = [x * 1.0 / sum(bb) for x in bb]


aa3 = []
cum = 0
for x, y in enumerate(aa2):
    if cum < 0.98:
        aa3.append(aa[x])
        cum += y
    else:
        break
th1 = aa3[-1]

bb3 = []
cum = 0
for x, y in enumerate(bb2):
    if cum < 0.98:
        bb3.append(bb[x])
        cum += y
    else:
        break
th2 = bb3[-1]

"""
good_labels1 = []
for x, y in Counter(km1.labels_).items():
    if y > th1:
        good_labels1.append(x)

good_labels2 = []
for x, y in Counter(km2.labels_).items():
    if y > th2:
        good_labels2.append(x)


print good_labels1
print good_labels2

total1 = 0
for x in good_labels1:
    total1 += Counter(km1.labels_)[x]

total2 = 0
for x in good_labels2:
    total2 += Counter(km2.labels_)[x]

print total1, total2, "kjslkfjs"


anchors_num1 = {}
for x in good_labels1:
    anchors_num1[x] = int(math.ceil(Counter(km1.labels_)[x] * 100.0 / total1))
    if anchors_num1[x] == 0:
        anchors_num1[x] = 1

anchors_num2 = {}
for x in good_labels2:
    anchors_num2[x] = int(math.ceil(Counter(km2.labels_)[x] * 100.0 / total2))
    if anchors_num2[x] == 0:
        anchors_num2[x] = 1


print anchors_num1
print anchors_num2



anchors1 = []
anchors2 = []

for x, y in anchors_num1.items():
    sample = np.random.choice(np.array(range(len(km1.labels_)))[km1.labels_ == x], size = y, replace = False)
    anchors1.extend(sample)

for x, y in anchors_num2.items():
    sample = np.random.choice(np.array(range(len(km2.labels_)))[km2.labels_ == x], size = y, replace = False)
    anchors2.extend(sample)


print anchors1
print anchors2


data11 = data1[anchors1, :]
data22 = data2[anchors2, :]
"""


# LLJLKJLKJLKJ
data11 = km1.cluster_centers_
data22 = km2.cluster_centers_





la = len(data11)
lb = len(data22)

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
    for j, _ in [(u, v) for u, v in xx[:top10percent] if v > 0.7]:
        graph.add_edge(nodes1[i], nodes2[j], weight=-distances[i][j])

isola = list(nx.isolates(graph))[:]
for iso in isola:
    graph.remove_node(iso)


print len(graph.nodes())
print len(graph.edges())




time1 = time.time()
mat = nx.max_weight_matching(graph, maxcardinality=True)
print time.time() - time1
matching = []
for x, y in mat:
    print x, y
    if x >= la:
        x, y = y, x
    matching.append((x, y - la))
matching = sorted(matching, key = lambda x: x[0])


total_distance = 0
for x, y in matching:
    print (x, y)
    total_distance += distances[x][y]
print total_distance


integration_vectors = {}
for x, y in matching:
    print pearsonr(data22[y, :], data11[x, :]), "PERASON"
    integration_vectors[x] = data22[y, :] - data11[x, :]


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

#np.savetxt("CLUES-corrected.csv", data1, fmt = "%.2f")
#np.savetxt("ImmVar-corrected.csv", data2, fmt = "%.2f")

np.savetxt("smartseq2-corrected.csv", data1, fmt = "%.2f")
np.savetxt("celseq2-corrected.csv", data2, fmt = "%.2f")

print "DONE"
