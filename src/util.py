import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def get_normal_sampler(mean = torch.zeros(2), sigma = torch.ones(2)):
    """
    Returns a function that draws from a normal distribution parametrized by specified parameters.
    Assume diagonal covariance, so sigma is a vector (not a matrix) which represents diagonal
    """
    
    d = mean.shape[0]
    
    def sampler(L = 1):
        """
        L is the number of data points we want to draw.
        """
        z = mean + torch.randn(L, d) * sigma
        logq = -0.5 * torch.sum(2 * torch.log(sigma) + np.log(2 * np.pi) + ((z - mean) / sigma) ** 2, 1)
        return z, logq
    
    return sampler


def plot_potential(U, ax):
    side = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(side, side)
    shape = X.shape
    X_flatten, Y_flatten = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))
    Z = np.concatenate([X_flatten, Y_flatten], 1)
    neg_logp = U(torch.Tensor(Z)).numpy()
    neg_logp = np.reshape(neg_logp, shape)
    p = np.exp(-neg_logp)
    p /= np.sum(p)
    Y = -Y # not sure why, but my plots are upside down compared to paper
    ax.pcolormesh(X, Y, p)
    
def plot_model_hist(model, ax, sampler = get_normal_sampler()):
    """
    We sample a bunch of points z0 ~ N(0, 1) and transform them through the normalizing flow, 
    to get an empirical distribution, which we then plot
    Method of plotting: divide the space into a bunch of blocks, then find empirical distribution 
    for each block.
    """
    side = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(side, side)
    counts = np.zeros(X.shape)
    p = np.zeros(X.shape)
    
    L = 100000 # batch of points to sample at once
    z_dim = 2
    print("Sampling", end='')
    for i in range(10):
        print('.', end='')
        z, logq = sampler(L)
        z, logq = Variable(z), Variable(logq)
        z_k, logq_k = model(z, logq)
        z_k, logq_k = z_k.data, logq_k.data
        q_k = torch.exp(logq_k)
        z_k = (z_k + 5) * 50
        for l in range(L):
            x, y = int(z_k[l, 1]), int(z_k[l, 0])
            if 0 <= x and x < 500 and 0 <= y and y < 500:
                counts[x, y] += 1
                p[x, y] += q_k[l]
    print()
    
    counts = np.maximum(counts, np.ones(counts.shape)) # no divide by zero, counts[x, y] == 0 iff p[x, y] == 0
    p /= counts
    p /= np.sum(p)
    Y = -Y # not sure why, but my plots are upside down compared to paper
    ax.pcolormesh(X, Y, p)
