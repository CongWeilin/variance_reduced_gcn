import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os

def name_trans(key, method):
    name_trans = {'sgcn':'{}'.format(method), 
                  'sgcn_plus':'{}+'.format(method), 
                  'sgcn_pplus':'{}++'.format(method), 
                  'sgcn_pplus_v2':'{}++(grad only)'.format(method), 
                  'full':'Full-batch'}
    return name_trans[key]

def smooth(scalars, weight=0.8):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def draw_train_loss_epoch(results, method):
    fig, axs = plt.subplots()
    for key, values in results.items():
        key = name_trans(key, method)
        loss_train, loss_test, loss_train_all, f1, grad_vars, best_val_index = values
        loss_train = loss_train[:best_val_index]
        loss_train = smooth(loss_train)
        x = np.arange(len(loss_train))
        axs.plot(x,loss_train,label=key)

    axs.set_xlabel('epoch')
    axs.set_ylabel('loss')
    axs.grid(True)

    plt.title('{} - {} - train_loss/epoch'.format(method, dataset))
    fig.tight_layout()
    plt.legend()
    plt.savefig('{}_train_loss_epoch.pdf'.format(prefix))

def draw_test_loss_epoch(results, method):
    fig, axs = plt.subplots()
    for key, values in results.items():
        key = name_trans(key, method)
        loss_train, loss_test, loss_train_all, f1, grad_vars, best_val_index = values
        loss_test = loss_test[:best_val_index]
        loss_test = smooth(loss_test)
        x = np.arange(len(loss_test))
        axs.plot(x,loss_test,label=key)

    axs.set_xlabel('epoch')
    axs.set_ylabel('loss')
    axs.grid(True)

    plt.title('{} - {} - test_loss/epoch'.format(method, dataset))
    fig.tight_layout()
    plt.legend()
    plt.savefig('{}_test_loss_epoch.pdf'.format(prefix))

def draw_train_loss_step(results, method):
    fig, axs = plt.subplots()
    for key, values in results.items():
        key = name_trans(key, method)
        loss_train, loss_test, loss_train_all, f1, grad_vars, best_val_index = values
        loss_train_all = loss_train_all[:400]
        loss_train_all = smooth(loss_train_all)
        x = np.arange(len(loss_train_all))
        axs.plot(x,loss_train_all,label=key)

    axs.set_xlabel('steps')
    axs.set_ylabel('loss')
    axs.grid(True)

    plt.title('{} - {} - train_loss/step'.format(method, dataset))
    fig.tight_layout()
    plt.legend()
    plt.savefig('{}_train_loss_step.pdf'.format(prefix))

def draw_grad_variance(results, method):
    fig, axs = plt.subplots()
    for key, values in results.items():
        key = name_trans(key, method)
        loss_train, loss_test, loss_train_all, f1, grad_vars, best_val_index = values
        grad_vars = grad_vars[:200]
        grad_vars = smooth(grad_vars)
        x = np.arange(len(grad_vars))
        axs.plot(x,grad_vars,label=key)

    axs.set_xlabel('steps')
    axs.set_ylabel('gradient variance')
    axs.grid(True)

    plt.title('{} - {} - variances/step'.format(method, dataset))
    fig.tight_layout()
    plt.legend()
    plt.savefig('{}_grad_vars_step.pdf'.format(prefix))

def get_f1_score(results, dataset, method):
    for key, values in results.items():
        key = name_trans(key, method)
        loss_train, loss_test, loss_train_all, f1, grad_vars, best_val_index = values
        print(dataset, method, key, '%.5f'%f1)

datasets = ['Flickr', 'Reddit', 'PPI', 'PPI-large', 'Yelp']
methods = ['LADIES', 'FastGCN', 'GraphSage', 'GraphSaint']
f_log = open('f1_scores.txt', 'w')

for dataset in datasets:
    for method in methods:
        prefix = '{}_{}'.format(method.lower(), dataset.lower())
        if os.path.exists('{}.pkl'.format(prefix)):
            with open('{}.pkl'.format(prefix),'rb') as f:
                results = pkl.load(f)
            draw_train_loss_epoch(results, method)
            draw_test_loss_epoch(results, method)
            draw_train_loss_step(results, method)
            get_f1_score(results, dataset, method, f_log)
            draw_grad_variance(results, method)
            
f_log.close()