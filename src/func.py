import torch
from torch.autograd import Variable
import numpy as np

# All the following functions take inputs of size (L, z_dim) and outputs size (L)

def w1(z):
    return torch.sin(2 * np.pi * z[:, 0] / 4)

def w2(z):
    return 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)

def w3(z):
    return 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

def U1(z):
    add1 = 0.5 * ((torch.norm(z, 2, 1) - 2) / 0.4) ** 2
    add2 = -torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2))
    return add1 + add2

def U2(z):
    return 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2

def U3(z):
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    return -torch.log(in1 + in2)

def U4(z):
    in1 = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    return -torch.log(in1 + in2)

def U_normal(z):
    """
    Negative log pdf of multivariate Gaussian with mean = 0, cov = I
    """
    
    return 0.5 * torch.sum(z ** 2, 1)

def get_U_normal(mean, cov):
    cov_inv = cov.inverse().unsqueeze(0)
    def U_normal2(z):
        z = z.unsqueeze(2)
        b_cov_inv = Variable(cov_inv.expand(z.shape[0], -1, -1))
        return 0.5 * torch.squeeze(z.permute(0, 2, 1).bmm(b_cov_inv).bmm(z))

    return U_normal2

