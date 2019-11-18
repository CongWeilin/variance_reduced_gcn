import matplotlib.pyplot as plt
from utils import *
import sys

dataset,layers = sys.argv[1], sys.argv[2]
sample_methods = ['full','ladies','fastgcn','graphsage','ladies-vr','fastgcn-vr','graphsage-vr']
fig, axs = plt.subplots()
for method in sample_methods:
    with open('{}.pkl'.format(method),'rb') as f:
        loss = pkl.load(f)
        loss = loss[:400]
        x = np.arange(len(loss))
        axs.plot(x,loss,label=method)
        
axs.set_xlabel('iters')
axs.set_ylabel('loss')
axs.grid(True)

fig.tight_layout()
plt.legend()
plt.savefig('train_loss_{}_{}.pdf'.format(dataset,layers))