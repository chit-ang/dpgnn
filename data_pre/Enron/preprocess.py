import itertools
from collections import defaultdict
from itertools import islice, chain
import re
from datetime import datetime
from datetime import timedelta


def lines_per_n(f, n):
    for line in f:
        yield ''.join(chain([line], itertools.islice(f, n - 1)))

node_data = defaultdict(lambda: ())
with open('enron_raw/vis.graph.nodeList.json') as f:
    for chunk in lines_per_n(f, 5):
        chunk = chunk.split("\n")
        id_string = chunk[1].split(":")[1]
        x = [x.start() for x in re.finditer('\"', id_string)]
        id = id_string[x[0] + 1:x[1]]

        name_string = chunk[2].split(":")[1]
        x = [x.start() for x in re.finditer('\"', name_string)]
        name = name_string[x[0] + 1:x[1]]

        idx_string = chunk[3].split(":")[1]
        x1 = idx_string.find('(')
        x2 = idx_string.find(')')
        idx = idx_string[x1 + 1:x2]

        print(id, name, idx)
        node_data[name] = (id, idx)

import dateutil.parser


def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    return d


links = []
ts = []
with open('enron_raw/vis.digraph.allEdges.json') as f:
    for chunk in lines_per_n(f, 5):
        chunk = chunk.split("\n")

        name_string = chunk[2].split(":")[1]
        x = [x.start() for x in re.finditer('\"', name_string)]
        from_id, to_id = name_string[x[0] + 1:x[1]].split("_")

        # gen = .split("_")
        # print (gen)
        # print (from_id, to_id)

        time_string = chunk[3].split("ISODate")[1]
        x = [x.start() for x in re.finditer('\"', time_string)]
        timestamp = getDateTimeFromISO8601String(time_string[x[0] + 1:x[1]])
        # timestamp= isodate.parse_datetime()
        # print (timestamp)
        ts.append(timestamp)
        links.append((from_id, to_id, timestamp))
        # print (node_data[from_id], node_data[to_id])
print(min(ts), max(ts))
print("# interactions", len(links))
links.sort(key=lambda x: x[2])

import networkx as nx
import numpy as np
SLICE_MONTHS = 2
START_DATE = min(ts)+ timedelta(200)
MAX_DATE = max(ts) - timedelta(200)

slices_links = defaultdict(lambda : nx.MultiGraph())
slices_features = defaultdict(lambda : {})

print ("Start date", START_DATE)
slice_id = 0
# Split the set of links in order by slices to create the graphs.
for (a, b, time) in links:
    prev_slice_id = slice_id

    datetime_object = time
    if datetime_object > MAX_DATE:
        months_diff = (MAX_DATE - START_DATE).days//30
    else:
        months_diff = (datetime_object - START_DATE).days//30
    slice_id = months_diff // SLICE_MONTHS
    slice_id = max(slice_id, 0)

    if slice_id == 1+prev_slice_id and slice_id > 0:
        slices_links[slice_id] = nx.MultiGraph()
        slices_links[slice_id].add_nodes_from(slices_links[slice_id-1].nodes(data=True))
        assert (len(slices_links[slice_id].edges()) ==0)
        #assert len(slices_links[slice_id].nodes()) >0

    if slice_id == 1+prev_slice_id and slice_id ==0:
        slices_links[slice_id] = nx.MultiGraph()

    if a not in slices_links[slice_id]:
        slices_links[slice_id].add_node(a)
    if b not in slices_links[slice_id]:
        slices_links[slice_id].add_node(b)
    slices_links[slice_id].add_edge(a,b, date=datetime_object)

for slice_id in slices_links:
    print ("# nodes in slice", slice_id, len(slices_links[slice_id].nodes()))
    print ("# edges in slice", slice_id, len(slices_links[slice_id].edges()))
    temp = np.identity(len(slices_links[max(slices_links.keys())].nodes()))

    slices_features[slice_id] = {}
    for idx, node in enumerate(slices_links[slice_id].nodes()):
        slices_features[slice_id][node] = temp[idx]

from scipy.sparse import csr_matrix


def remap(slices_graph, slices_features):
    all_nodes = []
    for slice_id in slices_graph:
        assert len(slices_graph[slice_id].nodes()) == len(slices_features[slice_id])
        all_nodes.extend(slices_graph[slice_id].nodes())
    all_nodes = list(set(all_nodes))
    print("Total # nodes", len(all_nodes), "max idx", max(all_nodes))
    ctr = 0
    node_idx = {}
    idx_node = []
    for slice_id in slices_graph:
        for node in slices_graph[slice_id].nodes():
            if node not in node_idx:
                node_idx[node] = ctr
                idx_node.append(node)
                ctr += 1
    slices_graph_remap = []
    slices_features_remap = []
    for slice_id in slices_graph:
        G = nx.MultiGraph()
        for x in slices_graph[slice_id].nodes():
            G.add_node(node_idx[x])
        for x in slices_graph[slice_id].edges(data=True):
            G.add_edge(node_idx[x[0]], node_idx[x[1]], date=x[2]['date'])
        assert (len(G.nodes()) == len(slices_graph[slice_id].nodes()))
        assert (len(G.edges()) == len(slices_graph[slice_id].edges()))
        slices_graph_remap.append(G)

    for slice_id in slices_features:
        features_remap = []
        for x in slices_graph_remap[slice_id].nodes():
            features_remap.append(slices_features[slice_id][idx_node[x]])
            # features_remap.append(np.array(slices_features[slice_id][idx_node[x]]).flatten())
        features_remap = csr_matrix(np.squeeze(np.array(features_remap)))
        slices_features_remap.append(features_remap)
    return (slices_graph_remap, slices_features_remap)

slices_links_remap, slices_features_remap = remap(slices_links, slices_features)
np.savez('Enron/graphs.npz', graph=slices_links_remap)