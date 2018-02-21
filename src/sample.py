"""
Module containing various samplers.
The interface of sampler is as follows: 
samplers are always functions with no parameters, which return 
a pair of a random sample, and the log probability of that sample.
"""

import torch
import numpy as np

def normal_sampler(mean = torch.zeros(2), sigma = torch.ones(2)):
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
