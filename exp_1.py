import pickle
import shutil
from pathlib import Path
from functools import reduce
import torch
import torch.optim
import torch.utils.data.sampler
from torch.nn import functional as F
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from neptune.new.types import File
import os
from os import path
import configs
from data.datamgr import SetDataManager
from methods.hypernets.fhypermaml import FHyperMAML
from methods.hypernets.bayeshmaml import BayesHMAML

from torchmetrics.classification import MulticlassCalibrationError

import backbone

from io_utils import model_dict, parse_args, get_best_file, setup_neptune

save_numeric_data = True

# plot uncertainty in classification
def plot_histograms(neptune_run, s1, s2, q1, q2, save_numeric_data=save_numeric_data):
    # seen support
    for i, scores in s1.items():
        if save_numeric_data:
            path = f'exp_1_data/Seen/Support/{i}'
            os.mkdir(path)
        scores = np.transpose(np.array(scores))
        for k, score in enumerate(scores):
            score = np.array(score)
            # print(f"score shape {score.shape}")
            fig = plt.figure()
            plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
            mu = np.mean(score)
            std = np.std(score)
            plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
            if neptune_run:
                neptune_run[f"Seen / Support / {i} / Class {k} histogram"].upload(File.as_image(fig))
            plt.close(fig)
            if save_numeric_data:
                if neptune_run:
                    neptune_run[f"Seen / Support / {i} / Class {k} data"].upload(File.as_pickle(score))
                filepath = path + f'/Class_{k}_data'
                with open(filepath, 'wb') as f:
                    pickle.dump(score, f)

    # seen query
    for i, scores in q1.items():
        if save_numeric_data:
            path = f'exp_1_data/Seen/Query/{i}'
            os.mkdir(path)
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
            # save on neptune
            if save_numeric_data:
                if neptune_run:
                    neptune_run[f"Seen / Query / {i} / Class {k} data"].upload(File.as_pickle(score))
                filepath = path + f'/Class_{k}_data'
                with open(filepath, 'wb') as f:
                    pickle.dump(score, f)

    # unseen support
    for i, scores in s2.items():
        if save_numeric_data:
            path = f'exp_1_data/Unseen/Support/{i}'
            os.mkdir(path)
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
            if save_numeric_data:
                # save on neptune
                neptune_run[f"Unseen / Support / {i} / Class {k} data"].upload(File.as_pickle(score))
                # save file locally
                filepath = path + f'/Class_{k}_data'
                with open(filepath, 'wb') as f:
                    pickle.dump(score, f)
    # unseen query
    for i, scores in q2.items():
        if save_numeric_data:
            path = f'exp_1_data/Unseen/Query/{i}'
            os.mkdir(path)
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
            if save_numeric_data:
                # save on neptune
                neptune_run[f"Unseen / Query / {i} / Class {k} data"].upload(File.as_pickle(score))
                # save file locally
                filepath = path + f'/Class_{k}_data'
                with open(filepath, 'wb') as f:
                    pickle.dump(score, f)


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


def initLocalDirectories():
    if path.isdir('exp_1_data'):
        shutil.rmtree('exp_1_data')
    os.mkdir('exp_1_data')
    os.mkdir('exp_1_data/Seen')
    os.mkdir('exp_1_data/Seen/Support')
    os.mkdir('exp_1_data/Seen/Query')
    os.mkdir('exp_1_data/Unseen')
    os.mkdir('exp_1_data/Unseen/Support')
    os.mkdir('exp_1_data/Unseen/Query')




def experiment(params_experiment):
    if save_numeric_data:
        initLocalDirectories()
    num_samples = params_experiment.num_samples
    base_file = configs.data_dir['omniglot'] + 'noLatin.json'
    val_file = configs.data_dir['emnist'] + 'val.json'

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

    if params_experiment.method == 'fhyper_maml':
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = FHyperMAML(model_dict[params_experiment.model], params=params_experiment,
                           approx=(params_experiment.method == 'maml_approx'),
                           **train_few_shot_params)
        
        if params_experiment.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.train_lr = 0.1
    elif params_experiment.method == 'bayes_hmaml':
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = BayesHMAML(model_dict[params_experiment.model], params=params_experiment,
                           approx=(params_experiment.method == 'maml_approx'),
                           **train_few_shot_params)
        
        model.weight_set_num_test = 5
        model.weight_set_num_train = 5

        if params_experiment.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.train_lr = 0.1
    else:
        raise ValueError('Experiment for fhyper_maml only')

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
    #neptune_run = None
    # primary batches for adaptation
    features = []
    labels = []

    x, labels1 = next(iter(val_loader))
    if labels:
        while reduce(np.intersect1d, (*labels, labels1)).size > 0:
            x, labels1 = next(iter(val_loader))
    features.append(x)
    labels.append(labels1)

    model.n_query = features[0].size(1) - model.n_support
    support_datas1 = []
    query_datas1 = []
    support_datas2 = []
    query_datas2 = []
    data = []
    model.train()

    # train on 'seen' data
    for i, x in enumerate(features):
        data.append(x)
        x = x.cuda()
        x_var = torch.autograd.Variable(x)
        support_data = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                             *x.size()[2:])  # support data
        query_data = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                           *x.size()[2:])  # query data
        support_datas1.append(support_data)
        query_datas1.append(query_data)

    features_unseen = []

    # new batch for experiment
    features2, labels2 = next(iter(val_loader))
    # print('finding val batch')
    # if there are repetitions between batches get another batch
    while reduce(np.intersect1d, (*labels, labels2)).size > 0:
        features2, labels2 = next(iter(val_loader))
    print(labels2)
    labels.append(labels2)
    features_unseen.append(features2)

    model.n_query = features[-1].size(1) - model.n_support
    for i, features2 in enumerate(features_unseen):
        # no tuning
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

    model.eval()
    # model.train()
    for num in range(num_samples):
        for k,weight in enumerate(model.classifier.parameters()):
            weight.fast=None
        if params_experiment.method == "fhyper_maml":
            model.manager.clear_all_fields()
        model.set_forward(data[0])

        for i, support_data1 in enumerate(support_datas1):
            if i not in s1:
                s1[i] = []
            s1[i].append(F.softmax(model(support_data1), dim=1)[0].clone().data.cpu().numpy())
        for i, query_data1 in enumerate(query_datas1):
            if i not in q1:
                q1[i] = []
            q1[i].append(F.softmax(model(query_data1), dim=1)[0].clone().data.cpu().numpy())
        if num % 25 == 0:
            print(f'Accuracy on query at: {num}/{num_samples}')
            query_data_labels = torch.from_numpy(np.repeat(range(n_way), n_query))
            topk_scores, topk_labels = model(query_data1).data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy().flatten()
            y_labels = query_data_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind == y_labels)
            task_accuracy = (top1_correct / len(query_data_labels)) * 100
            print(task_accuracy)
        for i, support_data2 in enumerate(support_datas2):
            if i not in s2:
                s2[i] = []
            s2[i].append(F.softmax(model(support_data2), dim=1)[0].clone().data.cpu().numpy())
        for i, query_data2 in enumerate(query_datas2):
            if i not in q2:
                q2[i] = []
            q2[i].append(F.softmax(model(query_data2), dim=1)[0].clone().data.cpu().numpy())
        if num % 25 == 0:
            query_data_labels = torch.from_numpy(np.repeat(range(n_way), n_query))
            topk_scores, topk_labels = model(query_data2).data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy().flatten()
            y_labels = query_data_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind == y_labels)
            task_accuracy = (top1_correct / len(query_data_labels)) * 100
            print(task_accuracy)
        
    plot_histograms(neptune_run, s1, s2, q1, q2)


def main():
    # params_experiment = parse_args('train')
    params_experiment = parse_args('experiment1')   # todo sprawdzic parametry i wywalic najlepiej
    experiment(params_experiment=params_experiment)


if __name__ == '__main__':
    main()
