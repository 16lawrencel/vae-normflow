import torch
import torch.optim as optim
from torch.autograd import Variable
import os

class Trainer_NormFlow:
    # wrapper over normalizing flow where we train
    def __init__(self, z_dim, model, sampler, save_path = None):
        """
        sampler is a function that admits no parameters and outputs a 
        random variate z and its corresponding log probability with 
        respect to some sampling distribution.
        Will probably be a Gaussian sampler.
        """
        
        self.z_dim = z_dim
        
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())
        self.sampler = sampler
        
        if save_path is not None: self.save_path = '../ckpts/' + save_path
        else: self.save_path = '../ckpts/normflow_z={}/'.format(z_dim)
    
    def save_model(self):
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        torch.save(self.model.state_dict(), self.save_path + 'model')
        torch.save(self.optimizer.state_dict(), self.save_path + 'optimizer')
    
    def load_model(self):
        if not os.path.exists(self.save_path): return
        self.model.load_state_dict(torch.load(self.save_path + 'model'))
        self.optimizer.load_state_dict(torch.load(self.save_path + 'optimizer'))
    
    def train(self, U, L = 1, num_epochs = 10, epoch_length = 10000):
        """
        Trains the normalizing flow to match p(z) = exp(-U(z)) / Z
        Note z0 ~ N(0, 1)
        U is assumed to be a function that takes input of size (L, z_dim) and output size (L)
        z_dim will be 2 in our case.
        """
        
        for epoch in range(1, num_epochs + 1):
            running_loss = 0
            for i in range(epoch_length):
                self.optimizer.zero_grad()
                
                z_0, logq_0 = self.sampler(L)
                z_0, logq_0 = Variable(z_0), Variable(logq_0)
                z_k, logq_k = self.model(z_0, logq_0)
                loss = torch.sum(U(z_k) + logq_k) / L
                running_loss += loss.data[0]
                
                loss.backward()
                self.optimizer.step()
            
            running_loss /= epoch_length
            print("Epoch {} Loss: {}".format(epoch, running_loss))
    