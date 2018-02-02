import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

class PlanarFlow(nn.Module):
    """
    Planar flow transformation, described in equations (10) - (12), (21) - (23) in https://arxiv.org/pdf/1505.05770.pdf
    We use tanh activation (h) function.
    """
    
    def __init__(self, z_dim):
        super(PlanarFlow, self).__init__()
        
        self.z_dim = z_dim
        self.h = nn.Tanh()
        
        self.u = Parameter(torch.randn(z_dim, 1) * 0.01)
        self.w = Parameter(torch.randn(z_dim, 1) * 0.01)
        self.b = Parameter(torch.randn(1, 1) * 0.01)
    
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
        m = -1 + torch.log(1 + torch.exp(x))
        u_h = self.u + (m - x) * self.w / (self.w.t().mm(self.w))
        
        logp = logp - torch.log(1 + psi.mm(u_h).squeeze())
        z = z + a.mm(u_h.t())
        
        return z, logp


class NormFlow(nn.Module):
    """
    Normalizing flow (composition of individual transforms), as described in equation (13) of the paper
    """
    def __init__(self, z_dim):
        super(NormFlow, self).__init__()
        
        self.z_dim = z_dim
        self.flows = nn.ModuleList()
    
    def add_flow(self, flow_type, K = 1):
        """
        Adds K copies of flows of type flow_type to the overall normalizing flow.
        flow_type can be either PlanarFlow or RadialFlow
        """
        for k in range(K):
            flow = flow_type(self.z_dim)
            self.flows.append(flow)
    
    def forward(self, z, logp):
        """
        Given a set of samples z and their respective log probabilities, returns 
        the corresponding values z_K = f_K ○ f_(K-1) ○ ... ○ f_1(z), as well as 
        the respective log probabilities log p_K(z_K).
        Sizes should be (L, z_dim) and (L), respectively.
        Outputs are the same size as the inputs.
        """
        
        for flow in self.flows: z, logp = flow(z, logp)
        return z, logp

    