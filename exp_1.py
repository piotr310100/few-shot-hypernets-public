from pathlib import Path
from tabnanny import check
from tkinter import S

import torch
import torch.optim
import torch.utils.data.sampler
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
from neptune.new.types import File

import configs
from data.datamgr import SetDataManager

from methods.hypernets.hypermaml import HyperMAML
from io_utils import model_dict, parse_args, get_best_file , get_assigned_file, setup_neptune
from methods.hypernets.utils import reparameterize


def experiment(params):
    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot, n_query=n_query)
    base_datamgr = SetDataManager(image_size, **train_few_shot_params)  # n_eposide=100
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        # params.model = 'Conv4S'

    if params.method == 'hyper_maml':
        model = HyperMAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'), **train_few_shot_params)
        if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(
        configs.save_dir,
        params.dataset,
        params.model,
        params.method
    )

    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    if params.checkpoint_suffix != "":
        checkpoint_dir = checkpoint_dir + "_" + params.checkpoint_suffix

    if params.dataset == "cross":
        if not Path(checkpoint_dir).exists():
            checkpoint_dir = checkpoint_dir.replace("cross", "miniImagenet")

    assert Path(checkpoint_dir).exists(), checkpoint_dir

    #modelfile   = get_resume_file(checkpoint_dir)

    params.checkpoint_dir = checkpoint_dir

    if not params.method in ['baseline', 'baseline++'] : 
        modelfile   = get_best_file(checkpoint_dir)
        
        print("Using model file", modelfile)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model.load_state_dict(tmp['state'])
        else:
            print("[WARNING] Cannot find 'best_file.tar' in: " + str(checkpoint_dir))

    model.train()

    x, _ = next(iter(base_loader))
    model.n_query = x.size(1) - model.n_support
    loss, loss_ce, loss_kld, loss_kld_no_scale, task_accuracy, sigma, mu = model.set_forward_loss(x, False)
    
    model.eval()
    model.weight_set_num_train = 1
    model.weight_set_num_test = 1
    x, _ = next(iter(base_loader))
    x = x.cuda()
    x_var = torch.autograd.Variable(x)
    support_data = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support, *x.size()[2:]) # support data
    query_data = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,  *x.size()[2:]) # query data\
    s = []
    q = []
    for _ in range(100):
        for weight in model.classifier.parameters():
            weight.fast = [reparameterize(weight.mu, weight.logvar)]
        s.append(F.softmax(model.forward(support_data), dim=1)[0].clone().data.cpu().numpy())
        q.append(F.softmax(model.forward(query_data), dim=1)[0].clone().data.cpu().numpy())
    s = np.array(s)
    q = np.array(q)

    neptune_run = setup_neptune(params)
    
    for k, col in enumerate(s.T):
        fig = plt.figure()
        plt.hist(col, edgecolor ="black")
        neptune_run[f"Support class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
    for k, col in enumerate(q.T):
        fig = plt.figure()
        plt.hist(col, edgecolor ="black")
        neptune_run[f"Query class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)

def main():        
   params = parse_args('train')
   experiment(params)

if __name__ == '__main__':
    main()