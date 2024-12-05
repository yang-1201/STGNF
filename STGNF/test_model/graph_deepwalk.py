import networkx as nx
from gensim.models import Word2Vec
import random
import numpy as np
import csv
import numpy as np
from scipy.sparse.linalg import eigs
import os
import sys
import torch

def get_adjacency_matrix(dataset):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data_store')
        data_path = os.path.join(data_path, 'PEMS04')
        data_path = os.path.join(data_path, 'distance.csv')
        # print(data_path)
        sys.path.append(data_path)
        num_of_vertices=307
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data_store/PEMS08/distance.csv')
        sys.path.append(data_path)
        num_of_vertices=170

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1]),float(i[2])) for i in reader]

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        dis=[float(i[2]) for i in reader]
    # A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
    #              dtype=np.float32)
    print(dis)
    std=np.std(np.array(dis))
    print(std)
    tu=[]
    for i, j,k in edges:

        tu.append((i, j, {'weight': np.exp(-(k*k) / (std*std))}))
        #tu.append((i, j))
    return tu
tu=get_adjacency_matrix('PEMSD8')
print(tu)

G = nx.Graph()

G.add_edges_from(tu)

def deepwalk_walk(G, walk_length, walk_num):

    walks = []
    for node in G.nodes():
        for i in range(walk_num):
            walk = [node]
            for j in range(walk_length):
                current=walk[-1]
                neighbors = list(G.neighbors(current))
                #print(current,neighbors)
                if len(neighbors) > 0:
                    weights=[G[current][neighbor]['weight'] for neighbor in neighbors]
                    next_node =random.choices(neighbors,weights=weights,k=1)[0]

                    walk.append(next_node)
                else:
                    break
            walks.append([str(i)  for i in walk])
    return walks
walks=deepwalk_walk(G, 10, 5)
print(walks)

model = Word2Vec(walks, vector_size=60, window=6, min_count=0, sg=1, workers=4,epochs=100)
print(model.wv['29'])
print(model.wv[0])
node_embeddings = {}
node_embeddings1=[]
for node in range(170):
    node_embeddings[str(node)] = model.wv[str(node)]
    node_embeddings1.append(model.wv[str(node)])
for index, word in enumerate(model.wv.key_to_index):
     print(index,word,model.wv[word])


import pickle
with open('../data_store/PEMS08/n_embedding_pems08.pkl',"wb") as file:
    pickle.dump(node_embeddings1,file)


with open('../data_store/PEMS08/n_embedding_pems08.pkl',"rb") as file:
    data=pickle.load(file)
print(torch.tensor(data).var(-1))