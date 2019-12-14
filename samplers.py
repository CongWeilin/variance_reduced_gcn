from utils import *

def fastgcn_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):    
    np.random.seed(seed)
    previous_nodes = batch_nodes
    sampled_nodes = [ ]
    adjs  = []
    pi = np.array(np.sum(lap_matrix_sq, axis=0))[0]
    p = pi / np.sum(pi)
    for d in range(depth):
        U = lap_matrix[previous_nodes , :]
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        sampled_nodes += [previous_nodes]
        previous_nodes = after_nodes
    sampled_nodes.reverse()
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes, sampled_nodes

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
    np.random.seed(seed)
    previous_nodes = batch_nodes
    sampled_nodes = [ ]
    adjs  = []
    for d in range(depth):
        U = lap_matrix[previous_nodes , :]
        pi = np.array(np.sum(lap_matrix_sq[previous_nodes , :], axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        sampled_nodes += [previous_nodes]
        previous_nodes = after_nodes
    sampled_nodes.reverse()
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes, sampled_nodes

def graphsage_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
    np.random.seed(seed)
    sampled_nodes = [ ]
    previous_nodes = batch_nodes
    adjs = []
    for d in range(depth):
        U = lap_matrix[previous_nodes , :]
        after_nodes = [previous_nodes]
        for U_row in U:
            indices = U_row.indices
            sampled_indices = np.random.choice(indices, samp_num_list[d], replace=True)
            after_nodes.append(sampled_indices)
        after_nodes = np.unique(np.concatenate(after_nodes))
        adj = U[:, after_nodes]
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        sampled_nodes.append(previous_nodes)
        previous_nodes = after_nodes
    adjs.reverse()
    sampled_nodes.reverse()
    return adjs, previous_nodes, batch_nodes, sampled_nodes

def graphsaint_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
    # random walk sampler with normalization
    previous_nodes = batch_nodes
    sampled = []
    all_nodes, all_edges = [previous_nodes], []
    for d in range(depth):
        U = lap_matrix[previous_nodes , :]
        after_nodes = []
        for i, U_row in enumerate(U):
            sample_indices = np.random.choice(U_row.indices, 1000, replace=True)
            after_nodes.append(sample_indices[0])
            for j in sample_indices:
                all_edges.append(np.sort([previous_nodes[i],j]))
            all_nodes.append(sample_indices)
        after_nodes = np.array(after_nodes)
        sampled.append(previous_nodes)
        previous_nodes = after_nodes

    sampled = np.unique(sampled)
    all_nodes = np.concatenate(all_nodes)

    u_edges, e_cnts = np.unique(all_edges, axis=0, return_counts=True)
    u_nodes, n_cnts = np.unique(all_nodes, return_counts=True)
    u_node_cnt = dict()
    for u_nodes_,n_cnts_ in zip(u_nodes, n_cnts):
        u_node_cnt[u_nodes_] = n_cnts_

    for u_edges_,e_cnts_ in zip(u_edges, e_cnts):
        i,j = u_edges_
        lap_matrix[i,j] *= float(u_node_cnt[i])/float(e_cnts_)
        lap_matrix[j,i] *= float(u_node_cnt[j])/float(e_cnts_)
    adj = lap_matrix[sampled,:][:,sampled]

    adjs = [sparse_mx_to_torch_sparse_tensor(adj) for d in range(depth)]
    sampled_nodes = [sampled for d in range(depth)]
    return adjs, sampled, sampled, sampled_nodes

def exact_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
    previous_nodes = batch_nodes
    sampled_nodes = [ ]
    adjs = []
    for d in range(depth):
        U = lap_matrix[previous_nodes, :]
        after_nodes = [previous_nodes]
        for U_row in U:
            indices = U_row.indices
            after_nodes.append(indices)
        after_nodes = np.unique(np.concatenate(after_nodes))
        adj = U[:, after_nodes]
        adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
        sampled_nodes.append(previous_nodes)
        previous_nodes = after_nodes
    adjs.reverse()
    sampled_nodes.reverse()
    return adjs, previous_nodes, batch_nodes, sampled_nodes

def full_batch_sampler(batch_nodes, num_nodes, lap_matrix, depth):
    adjs = [sparse_mx_to_torch_sparse_tensor(lap_matrix) for _ in range(depth)]
    input_nodes = np.arange(num_nodes)
    sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
    return adjs, input_nodes, sampled_nodes