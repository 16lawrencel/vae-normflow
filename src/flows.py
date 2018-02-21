import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.init import xavier_normal
import numpy as np

class PlanarFlow(nn.Module):
    """
    Planar flow transformation, described in equations (10) - (12), (21) - (23) in https://arxiv.org/pdf/1505.05770.pdf
    We use tanh activation (h) function.
    """
    
    def __init__(self, *, z_dim = 2):
        super(PlanarFlow, self).__init__()
        
        self.z_dim = z_dim
        self.h = nn.Tanh()
        self.m = nn.Softplus()
        
        self.u = Parameter(torch.randn(z_dim, 1) * z_dim)
        self.w = Parameter(torch.randn(z_dim, 1) * z_dim)
        self.b = Parameter(torch.randn(1, 1))
    
    def forward(self, z, logp):
        """
        Given a set of samples z and their respective log probabilities, returns 
        z' = f(z) and log p(z'), as described by the equations in the paper.
        Sizes should be (L, z_dim) and (L), respectively.
        Outputs are the same size as the inputs.
        """
        
        a = self.h(torch.mm(z, self.w) + self.b)
        psi = (1 - a ** 2).mm(self.w.t()) # derivative of tanh(x) is 1 - tanh^2(x)
        
        # see end of section A.1
        x = self.w.t().mm(self.u)
        m = -1 + self.m(x)
        u_h = self.u + (m - x) * self.w / (self.w.t().mm(self.w))
        
        logp = logp - torch.log(1 + psi.mm(u_h).squeeze())
        z = z + a.mm(u_h.t())
        
        return z, logp

class IAF(nn.Module):
    """
    Inverse Autoregressive Flow with MADE autoregressive network, as described here:
    https://arxiv.org/pdf/1606.04934.pdf
    https://arxiv.org/pdf/1502.03509.pdf
    """
    
    def __init__(self, *, z_dim = 2, h_dim = 0, hiddens = [10, 10]):
        super(IAF, self).__init__()
        
        self.z_dim = z_dim
        self.h_dim = h_dim
        
        self.w_list = nn.ParameterList()
        self.b_list = nn.ParameterList()
        self.masks = []
        last_hidden = z_dim + h_dim
        for hidden_dim in hiddens:
            self.w_list.append(Parameter(xavier_normal(torch.Tensor(hidden_dim, last_hidden))))
            self.b_list.append(Parameter(torch.zeros(hidden_dim)))
            self.masks.append(Variable(torch.zeros(last_hidden)))
            last_hidden = hidden_dim
        
        self.masks.append(Variable(torch.zeros(last_hidden)))
        
        # d is direct connection from input to output
        self.w_m = Parameter(xavier_normal(torch.Tensor(z_dim, last_hidden)))
        self.d_m = Parameter(xavier_normal(torch.Tensor(z_dim, z_dim + h_dim)))
        self.b_m = Parameter(torch.ones(z_dim))
        self.w_s = Parameter(xavier_normal(torch.Tensor(z_dim, last_hidden)))
        self.d_s = Parameter(xavier_normal(torch.Tensor(z_dim, z_dim + h_dim)))
        self.b_s = Parameter(torch.ones(z_dim))

    def forward(self, z, logp, h = None):
        self.masks[0] = Variable(torch.arange(0, self.z_dim).type(torch.LongTensor))
        if self.h_dim > 0:
            self.masks[0] = torch.cat(self.masks[0], torch.zeros(self.h_dim))

        for i in range(1, len(self.masks)):
            st = np.asscalar(np.min(self.masks[i - 1].data.numpy()))
            nd = self.z_dim - 1
            weights = torch.ones(nd - st) / (nd - st)
            self.masks[i] = Variable(torch.multinomial(weights, len(self.masks[i]), replacement = True)) + st
        
        if h is None: cur = z
        else: cur = torch.cat(z, h, dim = 1)
        init_cur = cur # will be used for direct connection

        M_tot = Variable(torch.eye(self.z_dim))

        for i in range(len(self.w_list)):
            w = self.w_list[i]
            b = self.b_list[i]
            dim1, dim2 = w.data.shape
            mask1 = self.masks[i + 1].unsqueeze(1).expand(dim1, dim2)
            mask2 = self.masks[i].unsqueeze(0).expand(dim1, dim2)
            M = torch.ge(mask1, mask2).type(torch.FloatTensor)
            M_tot = M @ M_tot
            
            cur = F.selu(b + cur.mm((M * w).t()))
        
        dim1, dim2 = self.w_m.data.shape
        mask1 = Variable(torch.arange(0, self.z_dim).type(torch.LongTensor)).unsqueeze(1).expand(dim1, dim2)
        mask2 = self.masks[-1].unsqueeze(0).expand(dim1, dim2)
        M = torch.gt(mask1, mask2).type(torch.FloatTensor)
        
        dim1, dim2 = self.d_m.data.shape
        mask1 = Variable(torch.arange(0, self.z_dim).type(torch.LongTensor)).unsqueeze(1).expand(dim1, dim2)
        mask2 = self.masks[0].unsqueeze(0).expand(dim1, dim2)
        M_d = torch.gt(mask1, mask2).type(torch.FloatTensor)
        
        m = self.b_m + cur.mm((M * self.w_m).t()) + init_cur.mm((M_d * self.d_m).t())
        s = self.b_s + cur.mm((M * self.w_s).t()) + init_cur.mm((M_d * self.d_s).t())
        s = torch.sigmoid(s)

        M_tot = M @ M_tot
        
        z = s * z + (1 - s) * m
        # reversing order of z, like paper does
        idx = torch.LongTensor(range(self.z_dim - 1, -1, -1))
        z = z[:, idx]
        logp = logp - torch.sum(torch.log(s), 1)
        
        return z, logp
    

class NormFlow(nn.Module):
    """
    Normalizing flow (composition of individual transforms), as described in equation (13) of the paper
    """
    def __init__(self, z_dim, *, flow_type = PlanarFlow, K = 0, name = None):
        """
        Can optionally add flows in constructor
        """
        super(NormFlow, self).__init__()
        
        self.z_dim = z_dim
        self.flows = nn.ModuleList()
        self.add_flow(flow_type, K)
        if name is None: self.name = "flow={}_K={}".format(flow_type.__name__, K)
        else: self.name = name
    
    def add_flow(self, flow_type, K = 1):
        """
        Adds K copies of flows of type flow_type to the overall normalizing flow.
        flow_type can be either PlanarFlow, RadialFlow (not implemented), or IAF
        """
        for k in range(K):
            flow = flow_type(z_dim = self.z_dim)
            self.flows.append(flow)
    
    def add_flow_object(self, flow):
        """
        Adds a specific instantiatized flow (so type is Flow object)
        """
        self.flows.append(flow)
    
    def forward(self, z, logp, h = None):
        """
        Given a set of samples z and their respective log probabilities, returns 
        the corresponding values z_K = f_K ○ f_(K-1) ○ ... ○ f_1(z), as well as 
        the respective log probabilities log p_K(z_K).
        Sizes should be (L, z_dim) and (L), respectively.
        Outputs are the same size as the inputs.
        """
        
        for flow in self.flows:
            if h is None: z, logp = flow(z, logp)
            else: z, logp = flow(z, logp, h)
        return z, logp
    
    def sample(self, sampler):
        """
        Given a sampler, returns a sample from the transformed distribution.
        """
        
        z, logp = sampler()
        return self(z, logp)
    
