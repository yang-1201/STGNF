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
        data_path = os.path.join('../../data_store/PEMS08/pems08.npz')
        sys.path.append(data_path)
        num_of_vertices=170

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A

def get_adjacency_matrix1(dataset):
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
        data_path = os.path.join('../data_store/PEMS08')
        data_path = os.path.join(data_path,"distance.csv")
        sys.path.append(data_path)
        num_of_vertices=170

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = torch.zeros(int(num_of_vertices), int(num_of_vertices))

    for i, j in edges:
        A[i, j] = 1
    for i in range(num_of_vertices):
        A[i,i]=1
    return A#A[:20,:20]
def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''
    print(W.shape)
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def build_edge_index(dataset,add_self_edges=True):
    source_nodes_ids, target_nodes_ids = [], []
    seen_edges = set()

    if dataset == 'PEMSD4':
        data_path = os.path.join('../data_store')
        data_path = os.path.join(data_path, 'PEMS04')
        data_path = os.path.join(data_path, 'distance.csv')
        # print(data_path)
        sys.path.append(data_path)
        num_of_vertices=307
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../../data_store/PEMS08/pems08.npz')
        sys.path.append(data_path)
        num_of_vertices=170
    else:
        raise ValueError

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    num_of_vertices=20
    for src_node, target_node in edges:
        if src_node <num_of_vertices and target_node <num_of_vertices:
        # if this edge hasn't been seen so far we add it to the edge index (coalescing - removing duplicates)
            if (src_node, target_node) not in seen_edges:  # it'd be easy to explicitly remove self-edges (Cora has none..)
                source_nodes_ids.append(src_node)
                target_nodes_ids.append(target_node)

                seen_edges.add((src_node, target_node))

    if add_self_edges:
        source_nodes_ids.extend(np.arange(num_of_vertices))
        target_nodes_ids.extend(np.arange(num_of_vertices))

    # shape = (2, E), where E is the number of edges in the graph
    edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))

    return edge_index

