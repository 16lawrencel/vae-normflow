import flows
import train
import sample

def train_U_model(U, model, *, sampler = sample.normal_sampler(), 
                  L = 100, num_epochs = 10, epoch_length = 1000, 
                  load_model = False, train_model = True, save_path = None):
    """
    Trains a model to approximate the probability distribution for U.
    """
    if save_path is None:
        save_path = "normflow_U={}_{}/".format(U.__name__, model.name)
    
    trainer = train.Trainer_NormFlow(model, sampler = sampler, save_path = save_path)
    if load_model: trainer.load_model()
    if train_model: trainer.train(U, L = L, num_epochs = num_epochs, epoch_length = epoch_length)
    return model, trainer

def train_U(U, *, z_dim = 2, flow_type = flows.PlanarFlow, K = 32, 
            sampler = sample.normal_sampler(), L = 100, num_epochs = 10, 
            epoch_length = 1000, load_model = False, train_model = True, save_path = None):
    
    model = flows.NormFlow(z_dim, flow_type = flow_type, K = K)
    return train_U_model(U, model, sampler = sampler, L = L, num_epochs = num_epochs, 
                         epoch_length = epoch_length, load_model = load_model, train_model = train_model, save_path = save_path)

