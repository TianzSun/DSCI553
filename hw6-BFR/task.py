import sys
from time import time
import random
from sklearn.cluster import KMeans
import numpy as np
import itertools


start = time()
"""
input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]

"""

input_file = "hw6_clustering.txt"
n_cluster = 5
output_file = "output.txt"
ans = ["The intermediate results:\n"]

data = list(map(lambda x: x.strip("\n").split(','), open(input_file).readlines()))
data = [(int(d[0]), tuple(list(map(lambda x: float(x), d[2:])))) for d in data]
id_data_dict = dict(data)
data_id_dict = dict(zip(list(id_data_dict.values()), list(id_data_dict.keys())))
data = list(map(lambda x: np.array(x), list(id_data_dict.values())))
random.shuffle(data)

# step 1:
size_20 = int(len(data) * 0.2)
data_use = data[:size_20]

# step 2:
kmeans = KMeans(n_clusters=n_cluster * 5).fit(data_use)

# step 3:

cluster_dict = {}
for label in kmeans.labels_:
    if label in cluster_dict:
        cluster_dict[label] += 1
    else:
        cluster_dict[label] = 1

RS_index = []
for label in cluster_dict:
    if cluster_dict[label] < 20:
        for i, x in enumerate(kmeans.labels_):
            if x == label:
                RS_index.append(i)
RS = []
for index in RS_index:
    RS.append(data_use[index])

for index in reversed(sorted(RS_index)):
    data_use.pop(index)

# step 4:

kmeans = KMeans(n_clusters=n_cluster).fit(data_use)

# step 5:

labeled_data = list(zip(kmeans.labels_, data_use))
DS = {}

for k_v in labeled_data:
    if k_v[0] not in DS:
        DS[k_v[0]] = {}
        DS[k_v[0]]["DATA"] = [data_id_dict[tuple(k_v[1])]]
        DS[k_v[0]]["N"] = 1
        DS[k_v[0]]["SUM"] = k_v[1]
        DS[k_v[0]]["SUMSQ"] = k_v[1] ** 2
    else:
        DS[k_v[0]]["DATA"].append(data_id_dict[tuple(k_v[1])])
        DS[k_v[0]]["N"] += 1
        DS[k_v[0]]["SUM"] += k_v[1]
        DS[k_v[0]]["SUMSQ"] += k_v[1] ** 2

# step 6:
kmeans = KMeans(n_clusters=len(RS) - 1).fit(RS)
cluster_dict = {}
for label in kmeans.labels_:
    if label in cluster_dict:
        cluster_dict[label] += 1
    else:
        cluster_dict[label] = 1

RS_2 = list(cluster_dict.keys())[list(cluster_dict.values()).index(2)]

RS_index = []
for key in list(cluster_dict.keys()):
    if key != RS_2:
        RS_index.append(list(kmeans.labels_).index(key))

labeled_data = tuple(zip(kmeans.labels_, RS))

CS = {}
for k_v in labeled_data:
    if k_v[0] == RS_2:
        if k_v[0] not in CS:
            CS[k_v[0]] = {}
            CS[k_v[0]]["DATA"] = [data_id_dict[tuple(k_v[1])]]
            CS[k_v[0]]["N"] = 1
            CS[k_v[0]]["SUM"] = k_v[1]
            CS[k_v[0]]["SUMSQ"] = k_v[1] ** 2
        else:
            CS[k_v[0]]["DATA"].append(data_id_dict[tuple(k_v[1])])
            CS[k_v[0]]["N"] += 1
            CS[k_v[0]]["SUM"] += k_v[1]
            CS[k_v[0]]["SUMSQ"] += k_v[1] ** 2


RS_old = RS[:]
RS = []
for index in reversed(sorted(RS_index)):
    RS.append(RS_old[index])
DS_num = 0
for c in DS:
    DS_num += DS[c]["N"]
CS_cluster = len(CS)
CS_num = 0
for c in CS:
    CS_num += CS[c]["N"]
RS_num = len(RS)

ans.append("Round 1: " + str(DS_num) + "," + str(CS_cluster) \
       + "," + str(CS_num) + "," + str(RS_num) + "\n")

for n in range(4):
    # step 7:
    if n < 3:
        data_use = data[size_20 * (n + 1):size_20 * (n + 2)]
    else:
        data_use = data[size_20 * 4:]
    # step 8:
    DS_index = set()
    for i in range(len(data_use)):
        point = data_use[i]
        distance_dict = {}
        for cluster in DS:
            centroid = DS[cluster]["SUM"] / DS[cluster]["N"]
            sig = DS[cluster]["SUMSQ"] / DS[cluster]["N"] - (DS[cluster]["SUM"] / DS[cluster]["N"]) ** 2
            z = (point - centroid) / sig
            distance_dict[cluster] = np.dot(z, z) ** 0.5
        m_distance = min(list(distance_dict.values()))
        for dc in distance_dict:
            if distance_dict[dc] == m_distance:
                cluster = dc
        if m_distance < (len(point) ** 0.5) * 2:
            # add point to DS
            DS[cluster]["DATA"].append(data_id_dict[tuple(point)])
            DS[cluster]["N"] += 1            
            DS[cluster]["SUM"] += point
            DS[cluster]["SUMSQ"] += point ** 2
            # add index to DS_index
            DS_index.add(i)
    # step 9:
    CS_index = set()
    for i in range(len(data_use)):
        if i not in DS_index:
            point = data_use[i]
            distance_dict = dict()
            for cluster in CS:
                centroid = CS[cluster]["SUM"] / CS[cluster]["N"]
                sig = CS[cluster]["SUMSQ"] / CS[cluster]["N"] - (CS[cluster]["SUM"] / CS[cluster]["N"]) ** 2
                z = (point - centroid) / sig
                distance_dict[cluster] = np.dot(z, z) ** 0.5
            m_distance = min(list(distance_dict.values()))
            for dc in distance_dict:
                if distance_dict[dc] == m_distance:
                    cluster = dc
            if m_distance < (len(point) ** 0.5) * 2:
                CS[cluster]["DATA"].append(data_id_dict[tuple(point)])
                CS[cluster]["N"] += 1
                CS[cluster]["SUM"] += point
                CS[cluster]["SUMSQ"] += point ** 2
                CS_index.add(i)
    # step 10:
    for i in range(len(data_use)):
        if i not in CS_index.union(DS_index):
            RS.append(data_use[i])
    # step 11:
    kmeans = KMeans(n_clusters=len(RS) - 1).fit(RS)
    cluster_dict = {}
    for label in kmeans.labels_:
        if label in cluster_dict:
            cluster_dict[label] += 1
        else:
            cluster_dict[label] = 1
    RS_2 = list(cluster_dict.keys())[list(cluster_dict.values()).index(2)]

    RS_index = []
    for key in list(cluster_dict.keys()):
        if key != RS_2:
            RS_index.append(list(kmeans.labels_).index(key))
    labeled_data = tuple(zip(kmeans.labels_, RS))
    CS = {}
    for k_v in labeled_data:
        if k_v[0] == RS_2:
            if k_v[0] not in CS:
                CS[k_v[0]] = {}
                CS[k_v[0]]["DATA"] = [data_id_dict[tuple(k_v[1])]]
                CS[k_v[0]]["N"] = 1
                CS[k_v[0]]["SUM"] = k_v[1]
                CS[k_v[0]]["SUMSQ"] = k_v[1] ** 2
            else:
                CS[k_v[0]]["DATA"].append(data_id_dict[tuple(k_v[1])])
                CS[k_v[0]]["N"] += 1
                CS[k_v[0]]["SUM"] += k_v[1]
                CS[k_v[0]]["SUMSQ"] += k_v[1] ** 2
    RS_old = RS[:]
    RS = []
    for index in reversed(sorted(RS_index)):
        RS.append(RS_old[index])

    # step 12:
    flag = True
    while True:
        compare = list(itertools.combinations(list(CS.keys()), 2))
        cluster_old = set(CS.keys())
        for cp in compare:
            centroid_1 = CS[cp[0]]["SUM"] / CS[cp[0]]["N"]
            centroid_2 = CS[cp[1]]["SUM"] / CS[cp[1]]["N"]
            sig_1 = CS[cp[0]]["SUMSQ"] / CS[cp[0]]["N"] - (CS[cp[0]]["SUM"] / CS[cp[0]]["N"]) ** 2
            sig_2 = CS[cp[1]]["SUMSQ"] / CS[cp[1]]["N"] - (CS[cp[1]]["SUM"] / CS[cp[1]]["N"]) ** 2
            z_1 = (centroid_1 - centroid_2) / sig_1
            z_2 = (centroid_1 - centroid_2) / sig_2
            m_1 = np.dot(z_1, z_1) ** 0.5
            m_2 = np.dot(z_2, z_2) ** 0.5
            m_distance = min(m_1, m_2)
            if m_distance < (len(CS[cp[0]]["SUM"]) ** 0.5) * 2:
                CS[cp[0]]["DATA"] = CS[cp[0]]["DATA"] + CS[cp[1]]["DATA"]
                CS[cp[0]]["N"] += CS[cp[1]]["N"]
                CS[cp[0]]["SUM"] += CS[cp[1]]["SUM"]
                CS[cp[0]]["SUMSQ"] += CS[cp[1]]["SUMSQ"]
                CS.pop(cp[1])
                flag = False
                break
        cluster_new = set(CS.keys())
        if cluster_new == cluster_old:
            break

    CS_key = list(CS.keys())
    if n == 3:
        for c_cs in CS_key:
            distance_dict = dict()
            for c_ds in DS:
                centroid_1 = DS[c_ds]["SUM"] / DS[c_ds]["N"]
                centroid_2 = CS[c_cs]["SUM"] / CS[c_cs]["N"]
                sig_1 = DS[c_ds]["SUMSQ"] / DS[c_ds]["N"] - (DS[c_ds]["SUM"] / DS[c_ds]["N"]) ** 2
                sig_2 = CS[c_cs]["SUMSQ"] / CS[c_cs]["N"] - (CS[c_cs]["SUM"] / CS[c_cs]["N"]) ** 2
                z_1 = (centroid_1 - centroid_2) / sig_1
                z_2 = (centroid_1 - centroid_2) / sig_2
                m_1 = np.dot(z_1, z_1) ** 0.5
                m_2 = np.dot(z_2, z_2) ** 0.5
                distance_dict[c_ds] = min(m_1, m_2)
            m_distance = min(list(distance_dict.values()))
            for dc in distance_dict:
                if distance_dict[dc] == m_distance:
                    cluster = dc
            if m_distance < (len(CS[c_cs]["SUM"]) ** 0.5) * 2:
                DS[cluster]["DATA"] = DS[cluster]["DATA"] + CS[c_cs]["DATA"]
                DS[cluster]["N"] += CS[c_cs]["N"]
                DS[cluster]["SUM"] += CS[c_cs]["SUM"]
                DS[cluster]["SUMSQ"] += CS[c_cs]["SUMSQ"]
                CS.pop(c_cs)
    DS_num = 0
    for c in DS:
        DS_num += DS[c]["N"]
    CS_cluster = len(CS)
    CS_num = 0
    for c in CS:
        CS_num += CS[c]["N"]
    RS_num = len(RS)

    ans.append("Round " + str(n+2) + ": " + str(DS_num) + "," + str(CS_cluster) \
               + "," + str(CS_num) + "," + str(RS_num) + "\n")

ans.append("\nThe clustering results:\n")
for cluster in DS:
    DS[cluster]["DATA"] = set(DS[cluster]["DATA"])
if CS:
    for cluster in CS:
        CS[cluster]["DATA"] = set(CS[cluster]["DATA"])

RS_set = set()
for point in RS:
    RS_set.add(data_id_dict[tuple(point)])

for point in range(len(id_data_dict)):
    if point in RS_set:
        ans.append(str(point) + ",-1\n")
    else:
        for cluster in DS:
            if point in DS[cluster]["DATA"]:
                ans.append(str(point) + "," + str(cluster) + "\n")
                break
        for cluster in CS:
            if point in CS[cluster]["DATA"]:
                ans.append(str(point) + ",-1\n")
                break

with open(output_file, "w+") as out_file:
    for line in ans:
        out_file.writelines(line)

end = time()
print(end-start)
