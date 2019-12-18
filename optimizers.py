from utils import *


def calculate_grad_variance(net, feat_data, labels, train_nodes, adjs_full):
    net_grads = []
    for p_net in net.parameters():
        net_grads.append(p_net.grad.data)
    clone_net = copy.deepcopy(net)
    _, _ = clone_net.calculate_loss_grad(feat_data, adjs_full, labels, train_nodes)

    clone_net_grad = []
    for p_net in clone_net.parameters():
        clone_net_grad.append(p_net.grad.data)
    del clone_net
    
    variance = 0.0
    for g1, g2 in zip(net_grads, clone_net_grad):
        variance += (g1-g2).norm(2) ** 2
    variance = torch.sqrt(variance)
    return variance

def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]


"""
SPIDER wrapper
"""


def spider_step(net, optimizer, feat_data, labels,
                train_nodes, valid_nodes,
                adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    Function to updated weights with a SPIDER backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    # record previous net mini batch gradient
    pre_net_mini = copy.deepcopy(net)

    # Compute full grad
    optimizer.zero_grad()
    current_loss, _ = net.calculate_loss_grad(
        feat_data, adjs_full, labels, train_nodes)

    # record previous net full gradient
    pre_net_full = []
    for p_net in net.parameters():
        pre_net_full.append(p_net.grad.data)

    #torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
    optimizer.step()

    running_loss = [current_loss.cpu().detach()]
    iter_num = 0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)
            # compute previous stochastic gradient
            pre_net_mini.zero_grad()
            # take backward
            pre_net_mini.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])

            # compute current stochastic gradient
            optimizer.zero_grad()
            current_loss = net.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])

            # take SCSG gradient step
            for p_net, p_mini, p_full in zip(net.parameters(), pre_net_mini.parameters(), pre_net_full):
                p_net.grad.data += p_full - p_mini.grad.data

            # only for experiment purpose to demonstrate ... 
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(net, feat_data, labels, train_nodes, adjs_full))

            # record previous net full gradient
            pre_net_full = []
            for p_net in net.parameters():
                pre_net_full.append(p_net.grad.data)

            # record previous net mini batch gradient
            del pre_net_mini
            pre_net_mini = copy.deepcopy(net)

            #torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance



"""
SGD wrapper
"""


def sgd_step(net, optimizer, feat_data, labels,
             train_nodes, valid_nodes,
             adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)

            # compute current stochastic gradient
            optimizer.zero_grad()
            current_loss = net.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes])

            # only for experiment purpose to demonstrate ... 
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(net, feat_data, labels, train_nodes, adjs_full))

            #torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance

"""
Full-batch
"""


def full_step(net, optimizer, feat_data, labels,
              train_nodes, valid_nodes,
              adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:

            # compute current stochastic gradient
        optimizer.zero_grad()
        current_loss, _ = net.calculate_loss_grad(
            feat_data, adjs_full, labels, train_nodes)

        # only for experiment purpose to demonstrate ... 
        if calculate_grad_vars:
            grad_variance.append(calculate_grad_variance(net, feat_data, labels, train_nodes, adjs_full))
        optimizer.step()

        # print statistics
        running_loss += [current_loss.cpu().detach()]
        iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance


"""
Used for Multi-level Spider
"""


class ForwardWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes):
        super(ForwardWrapper, self).__init__()
        self.n_layers = n_layers
        self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def forward_mini(self, net, staled_net, x, adjs, sampled_nodes):
        cached_outputs = []
        for ell in range(self.n_layers):
            stale_x = x if ell == 0 else self.hiddens[ell -
                                                      1, sampled_nodes[ell-1]].to(x)
            stale_x = staled_net.gcs[ell](stale_x, adjs[ell])
            stale_x = staled_net.dropout(staled_net.relu(stale_x))
            x = net.gcs[ell](x, adjs[ell])
            x = net.dropout(net.relu(x))
            x = x + self.hiddens[ell, sampled_nodes[ell]
                                 ].to(x) - stale_x.detach()
            cached_outputs.append(x.cpu().detach())
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        for ell in range(self.n_layers):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
        return x

    # do not update hiddens, this is the most brilliant coding trick in this file !!!
    def forward_mini_staled(self, staled_net, x, adjs, sampled_nodes):
        for ell in range(self.n_layers):
            x = staled_net.gcs[ell](x, adjs[ell])
            x = staled_net.dropout(staled_net.relu(x))
            x = x - x.detach() + self.hiddens[ell, sampled_nodes[ell]].to(x)
        x = staled_net.linear(x)
        x = F.log_softmax(x, dim=1)
        return x

    def forward_full(self, net, x, adjs, sampled_nodes):
        for ell in range(self.n_layers):
            x = net.gcs[ell](x, adjs[ell])
            x = net.relu(x)
            x = net.dropout(x)
            self.hiddens[ell, sampled_nodes[ell]] = x.cpu().detach()
        x = net.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


"""
Multi-level Variance Reduction
"""


def multi_level_spider_step_v2(net, optimizer, feat_data, labels,
                               train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                               train_data, inner_loop_num, forward_wrapper, device, dist_bound=1e-3, calculate_grad_vars=False):
    """
    Function to updated weights with a Multi-Level SPIDER backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    # record previous net mini batch gradient
    pre_net_mini = copy.deepcopy(net)

    # Compute full grad
    optimizer.zero_grad()
    outputs = forward_wrapper.forward_full(
        net, feat_data, adjs_full, sampled_nodes_full)
    current_loss = F.nll_loss(outputs[train_nodes], labels[train_nodes])
    current_loss.backward()
    # record previous net full gradient
    pre_net_full = []
    for p_net in net.parameters():
        pre_net_full.append(p_net.grad.data)

    initial_hiddens = copy.deepcopy(forward_wrapper.hiddens)
    # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
    optimizer.step()

    running_loss = [current_loss.cpu().detach()]
    iter_num = 0

    grad_variance = []

    # Run over the train_loader
    interupt = False
    while iter_num < inner_loop_num and not interupt:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)
            # compute previous stochastic gradient
            pre_net_mini.zero_grad()
            outputs = forward_wrapper.forward_mini_staled(
                pre_net_mini, feat_data[input_nodes], adjs, sampled_nodes)
            staled_loss = F.nll_loss(outputs, labels[output_nodes])
            # debug only, set mini-batch size as full batch, it should have the exact same curve as full-batch GD
            # outputs = forward_wrapper.forward_mini_staled(
            #     pre_net_mini, feat_data, adjs_full, sampled_nodes_full)
            # staled_loss = F.nll_loss(outputs[train_nodes], labels[train_nodes])
            staled_loss.backward()

            pre_net_mini_grad = []
            for p_mini in pre_net_mini.parameters():
                pre_net_mini_grad.append(p_mini.grad.data)

            # compute current stochastic gradient
            pre_net_mini.zero_grad()
            optimizer.zero_grad()

            outputs = forward_wrapper.forward_mini(
                net, pre_net_mini, feat_data[input_nodes], adjs, sampled_nodes)
            # debug only, set mini-batch size as full batch, it should have the exact same curve as full-batch GD
            # outputs = forward_wrapper.forward_mini(
            #     net, pre_net_mini, feat_data, adjs_full, sampled_nodes_full)

            # make sure the aggregated hiddens not too far
            current_hiddens = copy.deepcopy(forward_wrapper.hiddens)
            dist = (current_hiddens-initial_hiddens).abs().mean()
            if dist > dist_bound:
                interupt = True
                break

            current_loss = F.nll_loss(outputs, labels[output_nodes])
            # debug only, set mini-batch size as full batch, it should have the exact same curve as full-batch GD
            # current_loss = F.nll_loss(outputs[train_nodes], labels[train_nodes])
            current_loss.backward()

            # take SCSG gradient step
            for p_net, p_mini, p_full in zip(net.parameters(), pre_net_mini_grad, pre_net_full):
                p_net.grad.data += p_full - p_mini

            # only for experiment purpose to demonstrate ... 
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(net, feat_data, labels, train_nodes, adjs_full))

            # record previous net full gradient
            pre_net_full = []
            for p_net in net.parameters():
                pre_net_full.append(p_net.grad.data)

            # record previous net mini batch gradient
            del pre_net_mini
            pre_net_mini = copy.deepcopy(net)

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance


def multi_level_spider_step_v1(net, optimizer, feat_data, labels,
                               train_nodes, valid_nodes, adjs_full, sampled_nodes_full,
                               train_data, inner_loop_num, forward_wrapper, device, dist_bound=1e-3, calculate_grad_vars=False):
    """
    Function to updated weights with a Multi-Level SPIDER backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    # record previous net mini batch gradient
    pre_net_mini = copy.deepcopy(net)

    # Compute full grad
    optimizer.zero_grad()
    outputs = forward_wrapper.forward_full(
        net, feat_data, adjs_full, sampled_nodes_full)
    current_loss = F.nll_loss(outputs[train_nodes], labels[train_nodes])
    current_loss.backward()

    initial_hiddens = copy.deepcopy(forward_wrapper.hiddens)
    # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
    optimizer.step()

    running_loss = [current_loss.cpu().detach()]
    iter_num = 0

    grad_variance = []
    # Run over the train_loader
    interupt = False
    while iter_num < inner_loop_num and not interupt:
        for adjs, input_nodes, output_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)
            # compute previous stochastic gradient and compute current stochastic gradient
            optimizer.zero_grad()

            outputs = forward_wrapper.forward_mini(
                net, pre_net_mini, feat_data[input_nodes], adjs, sampled_nodes)

            # make sure the aggregated hiddens not too far
            current_hiddens = copy.deepcopy(forward_wrapper.hiddens)
            dist = (current_hiddens-initial_hiddens).abs().mean()
            if dist > dist_bound:
                interupt = True
                break

            current_loss = F.nll_loss(outputs, labels[output_nodes])
            # If we make mini-batch size = full batch, we want the algorithm act like full-batch GD !
            # outputs = forward_wrapper.forward_mini(net, pre_net_mini, feat_data, adjs_full, sampled_nodes_full)
            # current_loss = F.nll_loss(outputs[train_nodes], labels[train_nodes])
            current_loss.backward()

            # record previous net mini batch gradient
            del pre_net_mini
            pre_net_mini = copy.deepcopy(net)

            # only for experiment purpose to demonstrate ... 
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(net, feat_data, labels, train_nodes, adjs_full))
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.2)
            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance
