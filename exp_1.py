from pathlib import Path
from functools import reduce
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
from io_utils import model_dict, parse_args, get_best_file, setup_neptune
from methods.hypernets.utils import reparameterize


def plot_mu_sigma(neptune_run, model, i):
    # get flattened mu and sigma
    sigma, mu = model._mu_sigma(True)
    # plotting to neptune
    if sigma is not None:
        for name, value in sigma.items():
            fig = plt.figure()
            plt.plot(value, 's')
            neptune_run[f"sigma / {i} / {name} / plot"].upload(File.as_image(fig))
            plt.close(fig)
            fig = plt.figure()
            plt.hist(value, edgecolor="black")
            neptune_run[f"sigma / {i} / {name} / histogram"].upload(File.as_image(fig))
            plt.close(fig)
    if mu is not None:
        for name, value in mu.items():
            fig = plt.figure()
            plt.plot(value, 's')
            neptune_run[f"mu / {i} / {name} / plot"].upload(File.as_image(fig))
            plt.close(fig)
            fig = plt.figure()
            plt.hist(value, edgecolor="black")
            neptune_run[f"mu / {i} / {name} / histogram"].upload(File.as_image(fig))
            plt.close(fig)


# plot uncertainty in classification
def plot_histograms(neptune_run, s1, s2, q1, q2):

    # seen support
    for i, scores in s1.items():
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            # print(f"score shape {score.shape}")
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            neptune_run[f"Seen / Support / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)

    # seen query
    for i, scores in q1.items():
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            neptune_run[f"Seen / Query / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)

    # unseen support
    for i, scores in s2.items():
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            neptune_run[f"Unseen / Support / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)

    # unseen query
    for i, scores in q2.items():
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            neptune_run[f"Unseen / Query / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)


def getCheckpointDir(params, configs):
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        configs.save_dir,
        params.dataset,
        params.model,
        params.method
    )

    if params.train_aug:
        checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    if params.checkpoint_suffix != "":
        checkpoint_dir = checkpoint_dir + "_" + params.checkpoint_suffix

    if params.dataset == "cross":
        if not Path(checkpoint_dir).exists():
            checkpoint_dir = checkpoint_dir.replace("cross", "miniImagenet")

    assert Path(checkpoint_dir).exists(), checkpoint_dir
    return checkpoint_dir


def experiment(params_experiment):
    num_samples = params_experiment.num_samples
    if params_experiment.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params_experiment.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[params_experiment.dataset] + 'base.json'
        val_file = configs.data_dir[params_experiment.dataset] + 'val.json'

    if 'Conv' in params_experiment.model:
        if params_experiment.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    n_query = max(1, int(16 * params_experiment.test_n_way / params_experiment.train_n_way))
    # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

    n_way = params_experiment.n_way
    train_few_shot_params = dict(n_way=n_way, n_support=params_experiment.n_shot, n_query=n_query)
    # base_datamgr = SetDataManager(image_size, **train_few_shot_params)  # n_eposide = 100
    # base_loader = base_datamgr.get_data_loader(base_file, aug=params_experiment.train_aug)

    test_few_shot_params = dict(n_way=n_way, n_support=params_experiment.n_shot, n_query=n_query)
    val_datamgr = SetDataManager(image_size, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    if params_experiment.dataset in ['omniglot', 'cross_char']:
        assert params_experiment.model == 'Conv4' and not params_experiment.train_aug, 'omniglot only support Conv4 without augmentation'

    if params_experiment.method == 'hyper_maml':
        model = HyperMAML(model_dict[params_experiment.model], params=params_experiment,
                          approx=(params_experiment.method == 'maml_approx'),
                          **train_few_shot_params)
        if params_experiment.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.train_lr = 0.1
    else:
        raise ValueError('Experiment for hyper_maml only')

    model = model.cuda()

    params_experiment.checkpoint_dir = getCheckpointDir(params_experiment, configs)

    modelfile = get_best_file(params_experiment.checkpoint_dir)  # load best from given model
    print("Using model file", modelfile)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
    else:
        print("[WARNING] Cannot find 'best_file.tar' in: " + str(params_experiment.checkpoint_dir))

    neptune_run = setup_neptune(params_experiment)
    # primary batches for adaptation
    features = []
    labels = []

    for _ in range(params_experiment.num_batches_seen):
        features1, labels1 = next(iter(val_loader))
        if labels:
            while reduce(np.intersect1d, (*labels, labels1)).size > 0:
                features1, labels1 = next(iter(val_loader))
        features.append(features1)
        labels.append(labels1)

    model.n_query = features[0].size(1) - model.n_support
    support_datas1 = []
    query_datas1 = []
    support_datas2 = []
    query_datas2 = []
    model.train()
    # train on 'seen' data
    for i, features1 in enumerate(features):
        _ = model.set_forward_loss(features1, False)
        plot_mu_sigma(neptune_run, model, i)
        features1 = features1.cuda()
        x_var = torch.autograd.Variable(features1)
        support_data = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                             *features1.size()[2:])  # support data
        query_data = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                           *features1.size()[2:])  # query data
        support_datas1.append(support_data)
        query_datas1.append(query_data)

    # only draw one set from weights distribution
    model.weight_set_num_train = 1
    model.weight_set_num_test = 1

<<<<<<< Updated upstream
    s1 = []
    q1 = []
    # new batch for experiment
    x2, out2 = next(iter(val_loader))
    while np.intersect1d(out1, out2).size > 0:
        x2, out2 = next(iter(val_loader))

    print(out1)
    print(out2)

    model.n_query = x2.size(1) - model.n_support
    x2 = x2.cuda()
    x2_var = torch.autograd.Variable(x2)
    support_data2 = x2_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                           *x2.size()[2:])  # support data
    query_data2 = x2_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                         *x2.size()[2:])  # query data\

=======
    features_unseen = []
    # new batches for experiment
    for _ in range(params_experiment.num_batches_unseen):
        features2, labels2 = next(iter(val_loader))
        print('finding val batch')
        # if there are repetitions between batches get another batch
        while reduce(np.intersect1d, (*labels, labels2)).size > 0:
            features2, labels2 = next(iter(val_loader))
        print(labels2)
        labels.append(labels2)
        features_unseen.append(features2)

    model.n_query = features[-1].size(1) - model.n_support
>>>>>>> Stashed changes
    model.eval()
    for i, features2 in enumerate(features_unseen):
        features2 = features2.cuda()
        x2_var = torch.autograd.Variable(features2)
        support_data2 = x2_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                               *features2.size()[2:])  # support data
        query_data2 = x2_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                             *features2.size()[2:])  # query data
        support_datas2.append(support_data2)
        query_datas2.append(query_data2)

    s1 = {}
    q1 = {}
    s2 = {}
    q2 = {}
    model.weight_set_num_train = 1
    model.weight_set_num_test = 1

    for _ in range(num_samples):
        for weight in model.classifier.parameters():
            weight.fast = [reparameterize(weight.mu, weight.logvar)]
<<<<<<< Updated upstream

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
        plt.hist(col, edgecolor="black", range=[0, 1], bins=25)
        mu = np.mean(col)
        std = np.std(col)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Support1 class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
    for k, col in enumerate(q1.T):
        fig = plt.figure()
        plt.hist(col, edgecolor="black", range=[0, 1], bins=25)
        mu = np.mean(col)
        std = np.std(col)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Query1 class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
    for k, col in enumerate(s2.T):
        fig = plt.figure()
        plt.hist(col, edgecolor="black", range=[0, 1], bins=25)
        mu = np.mean(col)
        std = np.std(col)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Support2 class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
    for k, col in enumerate(q2.T):
        fig = plt.figure()
        plt.hist(col, edgecolor="black", range=[0, 1], bins=25)
        mu = np.mean(col)
        std = np.std(col)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Query2 class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
=======
        for i, support_data1 in enumerate(support_datas1):
            if i not in s1:
                s1[i] = []
            s1[i].append(F.softmax(model(support_data1), dim=1)[0].clone().data.cpu().numpy())
        for i, query_data1 in enumerate(query_datas1):
            if i not in q1:
                q1[i] = []
            q1[i].append(F.softmax(model(query_data1), dim=1)[0].clone().data.cpu().numpy())
        for i, support_data2 in enumerate(support_datas2):
            if i not in s2:
                s2[i] = []
            s2[i].append(F.softmax(model(support_data2), dim=1)[0].clone().data.cpu().numpy())
        for i, query_data2 in enumerate(query_datas2):
            if i not in q2:
                q2[i] = []
            q2[i].append(F.softmax(model(query_data2), dim=1)[0].clone().data.cpu().numpy())

    plot_histograms(neptune_run, s1, s2, q1, q2)
>>>>>>> Stashed changes


def main():
    # params_experiment = parse_args('train')
    params_experiment = parse_args('experiment1')
    experiment(params_experiment=params_experiment)


if __name__ == '__main__':
    main()
