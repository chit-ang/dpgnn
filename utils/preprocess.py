from __future__ import print_function
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
from tqdm import tqdm
import math
import sys
import torch
sys.path.append(r'D:\git\LDP-DG\utils')
from utilities import run_random_walks_n2v
import pickle as pkl
from scipy.stats import bernoulli
flags = tf.app.flags
FLAGS = flags.FLAGS
num_time_steps = FLAGS.time_steps
np.random.seed(123)
variance = 100000
def load_graphs_npz(dataset_str):
    graphs = np.load("data/{}/{}".format(dataset_str, "graphs.npz"), allow_pickle=True)['graph']
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    return graphs, adjs

def load_graphs(dataset_str):
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print(graphs)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    return graphs, adjs

#读无权图
def load_graphs_Unweighted(dataset_str):
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print(graphs)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    g=[]
    adjs_n=[]
    for i in range(len(graphs)):
        tmp=np.array(adjs[i].todense())
        tmp=np.int64(tmp > 0)
        gg=nx.from_numpy_matrix(tmp)
        g.append(gg)
        adjs_n.append(nx.adjacency_matrix(gg))
    return g, adjs_n

#baseline
def geom_prob_mass(eps, sensitivity=1):
    a = np.exp(-eps/sensitivity)
    weights = np.array(range(-variance, variance))
    prob_mass = []
    for w in weights:
        m = ((1 - a)/(1 + a))*(a** np.absolute(w) )
        prob_mass.append(m)
    return prob_mass

def geometric_mechanism(arr, eps, sensitivity):
    p = 1 - np.exp(-eps/sensitivity)
    z = np.random.geometric(p, len(arr)) - np.random.geometric(p, len(arr))
    noisy_arr = arr + z
    return np.array(noisy_arr)

def geometric(graphs,adjs,e):
    g=[]
    adjs_n=[]
    for i in range(len(graphs)):
        # edges= nx.adjacency_matrix(graphs[i]).toarray()
        edges = sp.coo_matrix(adjs[i])
        indices = np.vstack((edges.row, edges.col))
        index = torch.tensor(indices)
        values = torch.tensor(edges.data, dtype=torch.float32)
        # edge_index = torch.sparse_coo_tensor(index, values, edges.shape)
        sensitivity = 1
        edges_w_noisy = geometric_mechanism(values, e, sensitivity)
        edges_w_noisy_clipped = np.clip(edges_w_noisy, 0, None)
        edge_index = torch.sparse_coo_tensor(index, edges_w_noisy_clipped, edges.shape)
        edge_index=edge_index.to_dense()
        edge_index = np.array(edge_index)
        gg = nx.from_numpy_matrix(edge_index)
        g.append(gg)
        adjs_n.append(nx.adjacency_matrix(gg))
        # new_edges = np.concatenate((edges[:, [0, 1]], np.array([edges_w_noisy_clipped]).T), axis=1)
    return g, adjs_n

def log_laplace_mechanism(arr, eps, sensitivity):
    lap = np.random.laplace(loc=0, scale=sensitivity/eps, size=len(arr))
    noisy_arr = np.around(np.clip( arr * np.exp(lap) , 1, max(arr) )).astype(int)
    return np.array(noisy_arr)

def exponential(graphs,adjs,e):
    levels_length = 100000

    g = []
    adjs_n = []
    for i in range(len(graphs)):
        # edges= nx.adjacency_matrix(graphs[i]).toarray()
        edges = sp.coo_matrix(adjs[i])
        indices = np.vstack((edges.row, edges.col))
        index = torch.tensor(indices)
        values = torch.tensor(edges.data, dtype=torch.float32)
        # edge_index = torch.sparse_coo_tensor(index, values, edges.shape)
        sensitivity = 1
        edges_w_noisy = log_laplace_mechanism(values, e, sensitivity)
        edges_w_noisy_clipped = np.clip(edges_w_noisy, 0, None)
        edge_index = torch.sparse_coo_tensor(index, edges_w_noisy_clipped, edges.shape)
        edge_index = edge_index.to_dense()
        edge_index = np.array(edge_index)
        gg = nx.from_numpy_matrix(edge_index)
        g.append(gg)
        adjs_n.append(nx.adjacency_matrix(gg))
        # new_edges = np.concatenate((edges[:, [0, 1]], np.array([edges_w_noisy_clipped]).T), axis=1)
    return g, adjs_n









#随机响应
def k_response(epsilon):
    if epsilon==np.inf:
        p=1
    else:
        p = np.e ** epsilon / (np.e ** epsilon + 1)
    print(p)
    if np.random.random() < p:
        return 0
    else:
        return 1

#拉普拉斯（权重）
def load_noise_graphs_lp(graph,adjs,epsilon):
    g=[]
    adjs_n= []
    for i in range(len(graph)):
        adj_tmp = adjs[i].todense()
        adj_tmp=np.array(adj_tmp)
        beta = 1/ epsilon
        n = np.random.laplace(loc=0, scale=beta, size=int((adj_tmp.shape[0]-2)*(adj_tmp.shape[1]-1)/2))
        k=0
        for a in range(0,adj_tmp.shape[0]-1):
            for b in range(a+1,adj_tmp.shape[1]-1):
                adj_tmp[a,b]=adj_tmp[a,b]+n[k]
                # adj_tmp[a,b]=np.around(np.clip( adj_tmp[a,b] * np.exp(n[k]) , 0, 1 )).astype(int)
                k=k+1
                adj_tmp[b,a]=adj_tmp[a,b]
        adj_tmp=np.clip(adj_tmp,a_min=0,a_max=None)
        gg=nx.from_numpy_matrix(adj_tmp)
        g.append (gg)
        adjs_n.append(nx.adjacency_matrix(gg))
    return g,adjs_n

def get_noise(noise_type, size, seed,eps):
    delta = 1e-5
    eps = 10
    sensitivity = 1
    np.random.seed(seed)
    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))
    return noise

def aug_random_walk(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (d_mat.dot(adj)).tocoo()



def load_noise(graph, adjs, ep1):
    g = []
    adjs_n = []
    for i in range(len(graph)):
        adj = adjs[i].todense()
        # adj = aug_random_walk(adj)
        n_nodes = adj.shape[0]
        n_edges = (np.count_nonzero(adj))/2

        N = n_nodes

        A = sp.tril(adj, k=-1)
        print('getting the lower triangle of adj matrix done!')

        if ep1!=np.inf:
            eps_1 = ep1 * 0.01
            eps_2 = ep1 - eps_1
            noise = get_noise(noise_type='laplace', size=(N, N), seed=42,eps=eps_2)
            noise *= np.tri(*noise.shape, k=-1, dtype=np.bool)

            A += noise
            print(f'adding noise to the adj matrix done!')

            n_edges_keep = n_edges + int(
                get_noise(noise_type='laplace', size=1, seed=42,eps=eps_1)[0])
            print(f'edge number from {n_edges} to {n_edges_keep}')
            a_r = A.A.ravel()
            n_edges_keep=int(n_edges_keep)

            n_splits = 2
            len_h = int(len(a_r) // n_splits)
            # len_h = len(a_r)
            ind_list = []
            for i in tqdm(range(n_splits - 1)):
                ind = np.argpartition(a_r[len_h * i:len_h * (i + 1)], -n_edges_keep)[-n_edges_keep:]
                ind_list.append(ind + len_h * i)
            # ind = np.argpartition(a_r, -n_edges_keep)[-n_edges_keep:]
            ind = np.argpartition(a_r[len_h * (n_splits - 1):], -n_edges_keep)[-n_edges_keep:]
            ind_list.append(ind + len_h * (n_splits - 1))
            # ind_list.append(ind)

            ind_subset = np.hstack(ind_list)
            a_subset = a_r[ind_subset]
            ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

            row_idx = []
            col_idx = []
            for idx in ind:
                idx = ind_subset[idx]
                row_idx.append(idx // N)
                col_idx.append(idx % N)
                assert (col_idx < row_idx)
            data_idx = np.ones(n_edges_keep, dtype=np.int32)

            mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
            mat=mat+mat.T
            mat=mat.toarray()
            gg = nx.from_numpy_matrix(mat)
        else:
            gg = nx.from_numpy_matrix(adj)
        g.append(gg)
        adjs_n.append(nx.adjacency_matrix(gg))
    return g, adjs_n



#高斯（权重）
def load_noise_graphs_gs(graph,adjs,epsilon):
    g=[]
    adjs_n= []
    for i in range(len(graph)):
        adj_tmp = adjs[i].todense()
        adj_tmp=np.array(adj_tmp)
        delta = 10e-5
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / epsilon
        n = np.random.normal(loc=0, scale=sigma, size=int((adj_tmp.shape[0])*(adj_tmp.shape[1]-1)/2))
        k=0
        for a in range(0,adj_tmp.shape[0]-1):
            for b in range(a+1,adj_tmp.shape[1]-1):
                adj_tmp[a,b]=adj_tmp[a,b]+n[k]
                k=k+1
                adj_tmp[b,a]=adj_tmp[a,b]
        adj_tmp=np.clip(adj_tmp,a_min=0,a_max=None)
        gg=nx.from_numpy_matrix(adj_tmp)
        g.append (gg)
        adjs_n.append(nx.adjacency_matrix(gg))
    return g,adjs_n

def k_response_deg(x,epsilon):
    if epsilon==np.inf:
        p=0
    else:
        tmp1=1 / (np.e ** epsilon + 1)
        tmp2=(np.e ** epsilon - 1)/(np.e ** epsilon + 1)
        p = tmp1+x*tmp2
    return p

#本文方法
def load_noise_graphs_deg(graph,adjs,epsilon_1,epsilon_2,deg):
    ep2 =epsilon_2
    g=[]
    adjs_n= []
    for i in range(num_time_steps):
        adj_tmp = adjs[i].todense()
        adj_tmp=np.array(adj_tmp)
        beta = 1/ epsilon_1
        deg_sum=sum(deg[i])
        for a in range(0,adj_tmp.shape[0]-1):
            m=0
            degg = deg[i][a]
            k=degg/deg_sum
            pp = k_response_deg(k, ep2)
            tmp = bernoulli.rvs(p=pp)
            if tmp==1:
                noise = np.random.laplace(loc=0, scale=beta, size=int(adj_tmp.shape[1]-a-1))
                for b in range(a+1, adj_tmp.shape[1] - 1):
                    adj_tmp[a, b] = int(adj_tmp[a, b] + noise[m])
                    m=m+1
                    adj_tmp[b, a] = adj_tmp[a, b]
        adj_tmp=np.clip(adj_tmp,a_min=0,a_max=None)
        gg=nx.from_numpy_matrix(adj_tmp)
        g.append (gg)
        adjs_n.append(nx.adjacency_matrix(gg))
    return g,adjs_n

#rr
def k_random_response(value, values, epsilon):
    if not isinstance(values, list):
        raise Exception("The values should be list")
    if value not in values:
        raise Exception("Errors in k-random response")
    if epsilon==np.inf:
        p=1
    else:
        p = np.e ** epsilon / (np.e ** epsilon + len(values) - 1)
    if np.random.random() < p:
        return value
    x = values[np.random.randint(low=0, high=len(values))]#返回0到len(values)的任一一个数
    return x

#随机响应
def load_noise_graphs_rr(graph,adjs,epsilon):
    g=[]
    adjs_n= []
    values = []
    values.append(0)
    values.append(1)
    for i in range(len(graph)):
        adj_tmp = adjs[i].todense()
        adj_tmp=np.array(adj_tmp)
        for a in range(0,adj_tmp.shape[0]):
            for b in range(a,adj_tmp.shape[1]):
                if a!=b:
                    adj_tmp[a,b] = k_random_response(adj_tmp[a,b], values, epsilon)
                else:
                    adj_tmp[a,b]=1
                adj_tmp[b,a] = adj_tmp[a,b]
        gg=nx.from_numpy_matrix(adj_tmp)
        g.append (gg)
        adjs_n.append(nx.adjacency_matrix(gg))
    return g,adjs_n

def load_feats(dataset_str):
    features = np.load("data/{}/{}".format(dataset_str, "features.npz"), allow_pickle=True)['feats']
    print("Loaded {} X matrices ".format(len(features)))
    return features


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_graph_gcn(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    # adj_=adj
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized
    # return sparse_to_tuple(adj_normalized)

def normalize_graph_gcn_2(adj):
    adj_ = sp.coo_matrix(adj)
    # adj_ = adj+sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)



def get_context_pairs_incremental(graph):
    return run_random_walks_n2v(graph, graph.nodes())

def get_context_pairs(graphs, num_time_steps):
    print("Computing training pairs ...")
    context_pairs_train = []
    for i in range(0, num_time_steps):
        #随机游走
        context_pairs_train.append(run_random_walks_n2v(graphs[i], graphs[i].nodes()))
    return context_pairs_train

def get_evaluation_data(adjs, num_time_steps):
    eval_idx = num_time_steps - 1#获取倒数第1张图
    next_adjs = adjs[eval_idx]#测试集下一个图即最后一张图
    print("Generating and saving eval data ....")
    #创建数据
    train_edges, train_edges_false,val_edges, val_edges_false,test_edges, test_edges_false= \
            create_data_splits(adjs[eval_idx], next_adjs,val_mask_fraction=0.2, test_mask_fraction=0.2)
    return train_edges, train_edges_false,val_edges, val_edges_false,test_edges, test_edges_false

def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.2):
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false

