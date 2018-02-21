import torch
import numpy as np
import sys
import flows
import func
import sample
import train
import trainutil

def train_flows(i_num, load_id = False):
    print("TRAINING IAF")
    flow_type = flows.IAF
    K = 32
    L = 10
    num_epochs = 100
    epoch_length = 1000#10000
    sampler = sample.normal_sampler(sigma = torch.ones(2) * 5)
    load_model = False

    save_path_id = "normflow_U=id_flow={}_K=1/".format(flow_type.__name__)
    model_id, _ = trainutil.train_U(func.U_normal, flow_type = flow_type, K = 1, load_model = True, train_model = False, save_path = save_path_id)

    #i_list = [3, 4, 1, 2]
    i_list = [i_num]
    #for i in range(1, 5):
    for i in i_list:
        U = getattr(func, "U%d" % i)
        model = flows.NormFlow(2, flow_type = flow_type, K = K)
        if load_id:
            for k in range(K):
                model.flows[k].load_state_dict(model_id.flows[0].state_dict())

        _, trainer = trainutil.train_U_model(U, model, sampler = sampler, 
                L = L, num_epochs = num_epochs, epoch_length = epoch_length, load_model = load_model)
        trainer.save_model()

def train_flows2(i_num, load_id = False):
    print("TRAINING PF")
    flow_type = flows.PlanarFlow
    K = 32
    L = 1000
    num_epochs = 1000
    epoch_length = 1000#10000
    sampler = sample.normal_sampler(sigma = torch.ones(2))
    load_model = True
    train_model = True

    save_path_id = "normflow_U=id_flow={}_K=1/".format(flow_type.__name__)
    model_id, _ = trainutil.train_U(func.U_normal, flow_type = flow_type, K = 1, load_model = True, train_model = False, save_path = save_path_id)

    i_list = [i_num]
    #for i in range(4, 5):
    for i in i_list:
        U = getattr(func, "U%d" % i)
        model = flows.NormFlow(2, flow_type = flow_type, K = K)
        if load_id:
            for k in range(K):
                model.flows[k].load_state_dict(model_id.flows[0].state_dict())

        _, trainer = trainutil.train_U_model(U, model, sampler = sampler, 
                L = L, num_epochs = num_epochs, epoch_length = epoch_length, load_model = load_model, train_model = train_model)
        trainer.save_model()

def train_id(flow_type):
    flow_type = flow_type
    K = 1
    L = 10000
    num_epochs = 30
    epoch_length = 1000
    sampler = sample.normal_sampler(sigma = torch.ones(2) * 3)
    load_model = False
    save_path = "normflow_U=id_flow={}_K={}/".format(flow_type.__name__, K)
    U = func.get_U_normal(torch.zeros(2), torch.Tensor([[9, 0], [0, 9]]))

    trainutil.train_U(U, flow_type = flow_type, K = K, sampler = sampler, 
            L = L, num_epochs = num_epochs, epoch_length = epoch_length, load_model = load_model, 
            save_path = save_path)

if len(sys.argv) > 2:
    num = int(sys.argv[1])
    i_num = int(sys.argv[2])
    if num == 1: train_flows(i_num, True)
    else: train_flows2(i_num)

