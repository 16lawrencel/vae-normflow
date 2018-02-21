import torch
import torch.optim as optim
import flows
from torch.autograd import Variable
import sample
import os

class Trainer_NormFlow:
    # wrapper over normalizing flow where we train
    def __init__(self, model, *, sampler = None, save_path = None):
        """
        sampler is a function that admits no parameters and outputs a 
        random variate z and its corresponding log probability with 
        respect to some sampling distribution.
        Will probably be a Gaussian sampler.
        """
        
        self.z_dim = model.z_dim
        self.model = model
        
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = 0.001)
        
        if sampler is None: sampler = sample.normal_sampler()
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
    
    def train(self, U, *, L = 1, num_epochs = 10, epoch_length = 10000, save_ckpts = True):
        """
        Trains the normalizing flow to match p(z) = exp(-U(z)) / Z
        Note z0 ~ N(0, 1)
        U is assumed to be a function that takes input of size (L, z_dim) and output size (L)
        z_dim will be 2 in our case.
        If save_ckpts is True, then saves the model after every epoch
        """
        
        flow_name = self.model.flows[0].__class__.__name__
        print("Training {}".format(U.__name__))
        for epoch in range(1, num_epochs + 1):
            running_loss = 0
            running_length = 0
            for i in range(epoch_length):
                self.optimizer.zero_grad()
                
                z_0, logq_0 = self.sampler(L)
                z_0, logq_0 = Variable(z_0), Variable(logq_0)
                z_k, logq_k = self.model(z_0, logq_0)
                U_z = U(z_k)
                
                U_z = torch.min(U_z, torch.ones_like(U_z) * 10000)
                U_z = torch.max(U_z, torch.ones_like(U_z) * (-10000))
                #print(U_z)
                #ind = (U_z < 1).data
                #ind_z = ind.unsqueeze(1).expand(-1, z_k.data.shape[1])

                #z_k = z_k[ind_z]
                #logq_k = logq_k[ind]
                #U_z = U_z[ind]
                """
                if not ind.data.all():
                    print("SKIPPING")
                    continue
                """

                #print(len(z_k))

                loss = torch.sum(U_z + logq_k) / len(logq_k.data)
                #print(loss)
                loss.backward()
                
                skip = False
                if loss.data[0] != loss.data[0]: skip = True
                for param in self.model.parameters():
                    if not param.requires_grad: continue
                    bad = (param.grad != param.grad)
                    # bad = (param.grad >= 100) # any are too big / nan
                    if bad.data.any(): # any are nan
                        skip = True
                        break
                        
                if skip:
                    #print("SKIPPING NAN")
                    continue        
                
                running_loss += loss.data[0]
                running_length += 1
                
                self.optimizer.step()
            
            print("{} {}: RUNNING LENGTH: {}".format(U.__name__, flow_name, running_length))
            if running_length == 0: continue
            running_loss /= running_length
            print("{} {}: Epoch {} Loss: {}".format(U.__name__, flow_name, epoch, running_loss))
            if save_ckpts: self.save_model()

