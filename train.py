"""
Import necessary packages
"""
from utils import *
import argparse
import multiprocessing as mp
from samplers import fastgcn_sampler, ladies_sampler, graphsage_sampler, exact_sampler, graphsaint_sampler, full_batch_sampler

"""
Dataset arguments
"""
parser = argparse.ArgumentParser(
    description='Training GCN on Large-scale Graph Datasets')

parser.add_argument('--dataset', type=str, default='flickr',
                    help='Dataset name: cora/citeseer/pubmed/flickr/reddit')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default=200,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default=10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default=10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of GCN layers')
# parser.add_argument('--n_stops', type=int, default=200,
#                     help='Stop after number of batches that f1 do not increase')
parser.add_argument('--samp_num', type=int, default=512,
                    help='Number of sampled nodes per layer (only for ladies & factgcn)')
parser.add_argument('--sample_method', type=str, default='ladies',
                    help='Sampled Algorithms: ladies/fastgcn/graphsage/graphsaint/exact')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate')
parser.add_argument('--cuda', type=int, default=1,
                    help='Avaiable GPU ID')

args = parser.parse_args()
print(args)

"""
Prepare devices
"""
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

"""
Prepare data using multi-process
"""


def prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth):
    jobs = []
    for _ in process_ids:
        batch_idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[batch_idx]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes,
                                            samp_num_list, num_nodes, lap_matrix, lap_matrix_sq, depth))
        jobs.append(p)
    return jobs


lap_matrix, labels, feat_data, train_nodes, valid_nodes, test_nodes = preprocess_data(
    args.dataset)
print("Dataset information")
print(lap_matrix.shape, labels.shape, feat_data.shape,
      train_nodes.shape, valid_nodes.shape, test_nodes.shape)

if type(feat_data) == sp.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device)
else:
    feat_data = torch.FloatTensor(feat_data).to(device)

"""
Setup datasets and models for training (multi-class use sigmoid+binary_cross_entropy, use softmax+nll_loss otherwise)
"""
if args.dataset in ['cora', 'citeseer', 'pubmed', 'flickr', 'reddit']:
    from model import GCN
    from optimizers import spider_step, multi_level_spider_step_v1, multi_level_spider_step_v2, sgd_step, full_step
    from optimizers import ForwardWrapper, package_mxl
    labels = torch.LongTensor(labels).to(device)
    num_classes = labels.max().item()+1
elif args.dataset in ['ppi', 'ppi-large', 'amazon', 'yelp']:
    from model_mc import GCN
    from optimizers_mc import spider_step, multi_level_spider_step_v1, multi_level_spider_step_v2, sgd_step, full_step
    from optimizers_mc import ForwardWrapper, package_mxl
    labels = torch.FloatTensor(labels).to(device)
    num_classes = labels.shape[1]

if args.sample_method == 'ladies':
    sampler = ladies_sampler
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
elif args.sample_method == 'fastgcn':
    sampler = fastgcn_sampler
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
elif args.sample_method == 'exact':
    sampler = exact_sampler
    samp_num_list = np.array([-1 for _ in range(args.n_layers)])  # never used
elif args.sample_method == 'graphsage':
    sampler = graphsage_sampler
    assert(args.n_layers == 2)
    samp_num_list = np.array([5, 5])  # as proposed in GraphSage paper
elif args.sample_method == 'graphsaint':
    sampler = graphsaint_sampler
    samp_num_list = np.array([-1 for _ in range(args.n_layers)])  # never used

"""
This is a zeroth-order and first-order variance reduced version of Stochastic-GCN++
"""


def sgcn_pplus(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device):

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    pool = mp.Pool(args.pool_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                        lap_matrix, lap_matrix_sq, args.n_layers)

    all_res = []
    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)

    print(susage)

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
        train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    # this stupid wrapper is only used for sgcn++
    forward_wrapper = ForwardWrapper(
        len(feat_data), args.nhid, args.n_layers, num_classes)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, susage.parameters()))

    loss_train = []
    loss_test = []
    grad_norm = []
    loss_train_all = []

    for epoch in np.arange(args.epoch_num):
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        # prepare next epoch train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)

        inner_loop_num = args.batch_num
        cur_train_loss, cur_train_loss_all = multi_level_spider_step_v2(susage, optimizer, feat_data, labels,
                                                                        train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                                                                        train_data, inner_loop_num, forward_wrapper, device, dist_bound=2e-4)
        loss_train_all.extend(cur_train_loss_all)

        # calculate validate loss
        susage.eval()
        # susage.zero_grad()
        # train_loss, train_grad_norm = susage.calculate_loss_grad(
        #     feat_data, adjs_full, labels, train_nodes)
        susage.zero_grad()
        val_loss, val_grad_norm = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)

        cur_test_loss, cur_grad_norm = val_loss, val_grad_norm

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)
        grad_norm.append(cur_grad_norm)
        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss,
              '| grad norm: %.8f' % cur_grad_norm,)

    return susage, loss_train, loss_test, loss_train_all


"""
This is a first-order variance reduced version of Stochastic-GCN++
"""


def sgcn_pplus_v2(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device):

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    pool = mp.Pool(args.pool_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                        lap_matrix, lap_matrix_sq, args.n_layers)

    all_res = []
    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)

    print(susage)

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
        train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    # optimizer = optim.SGD(filter(lambda p : p.requires_grad, susage.parameters()), lr=0.7)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, susage.parameters()))

    loss_train = []
    loss_test = []
    grad_norm = []
    loss_train_all = []

    for epoch in np.arange(args.epoch_num):
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        # prepare next epoch train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)

        inner_loop_num = args.batch_num
        cur_train_loss, cur_train_loss_all = spider_step(susage, optimizer, feat_data, labels,
                                                         train_nodes, valid_nodes,
                                                         adjs_full, train_data, inner_loop_num, device)
        loss_train_all.extend(cur_train_loss_all)

        # calculate test loss
        susage.eval()
        # susage.zero_grad()
        # train_loss, train_grad_norm = susage.calculate_loss_grad(
        #     feat_data, adjs_full, labels, train_nodes)
        susage.zero_grad()
        val_loss, val_grad_norm = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)

        cur_test_loss, cur_grad_norm = val_loss, val_grad_norm

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)
        grad_norm.append(cur_grad_norm)
        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss,
              '| grad norm: %.8f' % cur_grad_norm)

    return susage, loss_train, loss_test, loss_train_all


"""
This is a zeroth-order variance reduced version of Stochastic-GCN+
"""


def sgcn_plus(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device):

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    pool = mp.Pool(args.pool_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                        lap_matrix, lap_matrix_sq, args.n_layers)

    all_res = []
    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)

    print(susage)

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
        train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    # this stupid wrapper is only used for sgcn++
    forward_wrapper = ForwardWrapper(
        len(feat_data), args.nhid, args.n_layers, num_classes)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, susage.parameters()))

    loss_train = []
    loss_test = []
    grad_norm = []
    loss_train_all = []

    for epoch in np.arange(args.epoch_num):
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        # prepare next epoch train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)

        inner_loop_num = args.batch_num
        # compare with sgcn_plus, the only difference is we use multi_level_spider_step_v1 here
        cur_train_loss, cur_train_loss_all = multi_level_spider_step_v1(susage, optimizer, feat_data, labels,
                                                                        train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                                                                        train_data, inner_loop_num, forward_wrapper, device, dist_bound=2e-4)
        loss_train_all.extend(cur_train_loss_all)

        # calculate validate loss
        susage.eval()
        # susage.zero_grad()
        # train_loss, train_grad_norm = susage.calculate_loss_grad(
        #     feat_data, adjs_full, labels, train_nodes)
        susage.zero_grad()
        val_loss, val_grad_norm = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)

        cur_test_loss, cur_grad_norm = val_loss, val_grad_norm

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)
        grad_norm.append(cur_grad_norm)
        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss,
              '| grad norm: %.8f' % cur_grad_norm,)

    return susage, loss_train, loss_test, loss_train_all


"""
This is just an unchanged Stochastic-GCN 
"""


def sgcn(feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device):

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    pool = mp.Pool(args.pool_num)
    lap_matrix_sq = lap_matrix.multiply(lap_matrix)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                        lap_matrix, lap_matrix_sq, args.n_layers)

    all_res = []
    susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=num_classes,
                 layers=args.n_layers, dropout=args.dropout).to(device)
    susage.to(device)

    print(susage)

    adjs_full, input_nodes_full, sampled_nodes_full = full_batch_sampler(
        train_nodes, len(feat_data), lap_matrix, args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, susage.parameters()))

    loss_train = []
    loss_test = []
    grad_norm = []
    loss_train_all = []

    for epoch in np.arange(args.epoch_num):
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        # prepare next epoch train data
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, samp_num_list, len(feat_data),
                            lap_matrix, lap_matrix_sq, args.n_layers)

        inner_loop_num = args.batch_num
        cur_train_loss, cur_train_loss_all = sgd_step(susage, optimizer, feat_data, labels,
                                                      train_nodes, valid_nodes,
                                                      adjs_full, train_data, inner_loop_num, device)
        # it can also run full-batch GD by ignoring all the samplings
        # cur_train_loss, cur_train_loss_all = full_step(susage, optimizer, feat_data, labels,
        #                         train_nodes, valid_nodes,
        #                         adjs_full, train_data, inner_loop_num, device)
        loss_train_all.extend(cur_train_loss_all)

        # calculate test loss
        susage.eval()
        # susage.zero_grad()
        # train_loss, train_grad_norm = susage.calculate_loss_grad(
        #     feat_data, adjs_full, labels, train_nodes)
        susage.zero_grad()
        val_loss, val_grad_norm = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)
        cur_test_loss, cur_grad_norm = val_loss, val_grad_norm

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)
        grad_norm.append(cur_grad_norm)
        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss,
              '| grad norm: %.8f' % cur_grad_norm,)

    return susage, loss_train, loss_test, loss_train_all


results = dict()

print('>>> sgcn')
susage, loss_train, loss_test, loss_traub_all = sgcn(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device)
results['sgcn'] = [loss_train, loss_test, loss_traub_all]

print('>>> sgcn_plus')
susage, loss_train, loss_test, loss_traub_all = sgcn_plus(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device)
results['sgcn_plus'] = [loss_train, loss_test, loss_traub_all]

print('>>> sgcn_pplus')
susage, loss_train, loss_test, loss_traub_all = sgcn_pplus(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device)
results['sgcn_pplus'] = [loss_train, loss_test, loss_traub_all]

print('>>> sgcn_pplus_v2')
susage, loss_train, loss_test, loss_traub_all = sgcn_pplus_v2(
    feat_data, labels, lap_matrix, train_nodes, valid_nodes, test_nodes,  args, device)
results['sgcn_pplus_v2'] = [loss_train, loss_test, loss_traub_all]

with open('result.pkl', 'wb') as f:
    pkl.dump(results, f)
