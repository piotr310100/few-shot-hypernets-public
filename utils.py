import torch
import numpy as np

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity) 


class FMetricsManager:
    """manager for metrics to report as output to stdout/ neptune.ai for the train phase"""
    def assert_exist(self, atrib_name):
        assert atrib_name in self.atribs, "invalid atrib name"
    def __init__(self,flow_w:float):
        self.acc = []
        self.loss = []
        self.loss_ce = []
        self.flow_loss = []
        self.flow_loss_scaled = []
        self.flow_density_loss = []
        self.flow_loss_raw = []
        self.theta_norm = []
        self.delta_theta_norm = []
        self.flow_w = flow_w
        self.atribs = ['acc', 'loss', 'loss_ce', 'flow_loss', 'flow_density_loss','flow_loss_raw', 'theta_norm', 'delta_theta_norm']

    def clear_field(self,atrib_name):
        self.assert_exist(atrib_name)
        getattr(self,atrib_name).clear()
    def get_metrics(self,clean_after:bool=True):
        for atrib in self.atribs:
            if not getattr(self,atrib):
                self.append(atrib,torch.tensor([0]).cuda())
        out = {'accuracy/train': np.asarray(self.acc).mean(),
                'loss': torch.stack(self.loss).mean(dtype=torch.float).item(),  # loss := loss_ce - flow_w * loss_flow
                'loss_ce':torch.stack(self.loss_ce).mean(dtype=torch.float).item(),
                'flow_loss': torch.stack(self.flow_loss).mean(dtype=torch.float).item(),    # loss_flow := flow_output_loss - density_loss (before scaling with flow_w)
                'flow_loss_scaled': torch.stack(self.flow_loss_scaled).mean(dtype=torch.float).item(),  # loss_flow * flow_w
                'flow_density_loss': torch.stack(self.flow_density_loss).mean(dtype=torch.float).item(),    # density component before scaling with flow_w
                'flow_loss_raw': torch.stack(self.flow_loss_raw).mean(dtype=torch.float).item(), # loss_flow before substracting density component (and before scaling with flow_w)
                'theta_norm': torch.stack(self.theta_norm).mean(dtype=torch.float).item(),
                'delta_theta_norm': torch.stack(self.delta_theta_norm).mean(dtype=torch.float).item()
                }
        if clean_after:
            for atrib in self.atribs:
                getattr(self,atrib).clear()

        return out
    def append(self, atrib_name, value):
        self.assert_exist(atrib_name)
        if type(value) is torch.Tensor:
            value = value.squeeze().cuda()
        if atrib_name == 'flow_loss' and self.flow_w is not None:
            self.flow_loss_scaled.append(value * self.flow_w)
        getattr(self, atrib_name).append(value)