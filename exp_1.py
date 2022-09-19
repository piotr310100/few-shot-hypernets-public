from pathlib import Path

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
from io_utils import model_dict, parse_args, get_best_file , setup_neptune
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

    test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=n_query)
    val_datamgr = SetDataManager(image_size, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

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

    neptune_run = setup_neptune(params)    

    x, out1 = next(iter(val_loader))
    model.n_query = x.size(1) - model.n_support
    loss, loss_ce, loss_kld, loss_kld_no_scale, task_accuracy, sigma, mu = model.set_forward_loss(x, False)
    sigma, mu = model._mu_sigma(True)
    if sigma is not None:
        for name, value in sigma.items():
            fig = plt.figure()
            plt.plot(value, 's')
            neptune_run[f"sigma / {name} / plot"].upload(File.as_image(fig))
            plt.close(fig)
            fig = plt.figure()
            plt.hist(value, edgecolor ="black")
            neptune_run[f"sigma / {name} / histogram"].upload(File.as_image(fig))
            plt.close(fig)
    if mu is not None:
        for name, value in mu.items():
            fig = plt.figure()
            plt.plot(value, 's')
            neptune_run[f"mu / {name} / plot"].upload(File.as_image(fig))
            plt.close(fig)
            fig = plt.figure()
            plt.hist(value, edgecolor ="black")
            neptune_run[f"mu / {name} / histogram"].upload(File.as_image(fig))
            plt.close(fig)

    x = x.cuda()
    x_var = torch.autograd.Variable(x)
    support_data = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support, *x.size()[2:]) # support data
    query_data = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,  *x.size()[2:]) # query data
    
    model.eval()
    model.weight_set_num_train = 1
    model.weight_set_num_test = 1

    s1 = []
    q1 = []
    x2, out2 = next(iter(val_loader))
    while np.intersect1d(out1, out2).size > 0:
        x2, out2 = next(iter(val_loader))

    print(out1)
    print(out2)

    model.n_query = x2.size(1) - model.n_support
    x2 = x2.cuda()
    x2_var = torch.autograd.Variable(x2)
    support_data2 = x2_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support, *x2.size()[2:]) # support data
    query_data2 = x2_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,  *x2.size()[2:]) # query data\
    
    model.eval()
    model.weight_set_num_train = 1
    model.weight_set_num_test = 1

    s2 = []
    q2 = []
    for _ in range(100):
        for weight in model.classifier.parameters():
            weight.fast = [reparameterize(weight.mu, weight.logvar)]

        s1.append(F.softmax(model(support_data), dim=1)[0].clone().data.cpu().numpy())
        q1.append(F.softmax(model(query_data), dim=1)[0].clone().data.cpu().numpy())
        
        s2.append(F.softmax(model(support_data2), dim=1)[0].clone().data.cpu().numpy())
        q2.append(F.softmax(model(query_data2), dim=1)[0].clone().data.cpu().numpy())

    s1 = np.array(s1)
    q1 = np.array(q1)
    s2 = np.array(s2)
    q2 = np.array(q2)

    for k, col in enumerate(s1.T):
        fig = plt.figure()
        plt.hist(col, edgecolor ="black", range=[0, 1], bins = 25)
        mu = np.mean(col)
        std = np.std(col)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Support1 class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
    for k, col in enumerate(q1.T):
        fig = plt.figure()
        plt.hist(col, edgecolor ="black", range=[0, 1], bins = 25)
        mu = np.mean(col)
        std = np.std(col)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Query1 class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
    for k, col in enumerate(s2.T):
        fig = plt.figure()
        plt.hist(col, edgecolor ="black", range=[0, 1], bins = 25)
        mu = np.mean(col)
        std = np.std(col)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Support2 class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
    for k, col in enumerate(q2.T):
        fig = plt.figure()
        plt.hist(col, edgecolor ="black", range=[0, 1], bins = 25)
        mu = np.mean(col)
        std = np.std(col)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Query2 class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
    

def main():        
   params = parse_args('train')
   experiment(params)

if __name__ == '__main__':
    main()