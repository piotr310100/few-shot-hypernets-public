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

from torchmetrics.functional.classification import multiclass_calibration_error as MCE

import backbone

from io_utils import model_dict, parse_args, get_best_file, setup_neptune

save_numeric_data = True

# plot uncertainty in classification
def plot_histograms(neptune_run, model_name, s1, s2, q1, q2, save_numeric_data=save_numeric_data):
    # seen support
    if save_numeric_data:
        path = f'exp_1_data/Seen/Support/{model_name}'
        os.mkdir(path)
    scores = np.transpose(np.array(s1))
    for k, score in enumerate(scores):
        score = np.array(score)
        # print(f"score shape {score.shape}")
        fig = plt.figure()
        plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
        mu = np.mean(score)
        std = np.std(score)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        if neptune_run:
            neptune_run[f"Seen / Support / {model_name} / Class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
        if save_numeric_data:
            if neptune_run:
                neptune_run[f"Seen / Support / {model_name} / Class {k} data"].upload(File.as_pickle(score))
            filepath = path + f'/Class_{k}_data'
            with open(filepath, 'wb') as f:
                pickle.dump(score, f)

    # seen query
    if save_numeric_data:
        path = f'exp_1_data/Seen/Query/{model_name}'
        os.mkdir(path)
    scores = np.transpose(np.array(q1))
    for k, score in enumerate(scores):
        score = np.array(score)
        fig = plt.figure()
        plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
        mu = np.mean(score)
        std = np.std(score)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Seen / Query / {model_name} / Class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
        # save on neptune
        if save_numeric_data:
            if neptune_run:
                neptune_run[f"Seen / Query / {model_name} / Class {k} data"].upload(File.as_pickle(score))
            filepath = path + f'/Class_{k}_data'
            with open(filepath, 'wb') as f:
                pickle.dump(score, f)

    # unseen support
    if save_numeric_data:
        path = f'exp_1_data/Unseen/Support/{model_name}'
        os.mkdir(path)
    scores = np.transpose(np.array(s2))
    for k, score in enumerate(scores):
        score = np.array(score)
        fig = plt.figure()
        plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
        mu = np.mean(score)
        std = np.std(score)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Unseen / Support / {model_name} / Class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
        if save_numeric_data:
            # save on neptune
            neptune_run[f"Unseen / Support / {model_name} / Class {k} data"].upload(File.as_pickle(score))
            # save file locally
            filepath = path + f'/Class_{k}_data'
            with open(filepath, 'wb') as f:
                pickle.dump(score, f)
    # unseen query
    if save_numeric_data:
        path = f'exp_1_data/Unseen/Query/{model_name}'
        os.mkdir(path)
    scores = np.transpose(np.array(q2))
    for k, score in enumerate(scores):
        score = np.array(score)
        fig = plt.figure()
        plt.hist(score, edgecolor="black", range=[0, 1], bins=25)
        mu = np.mean(score)
        std = np.std(score)
        plt.title(f'$\mu = {mu:.3}, \sigma = {std:.3}$')
        neptune_run[f"Unseen / Query / {model_name} / Class {k} histogram"].upload(File.as_image(fig))
        plt.close(fig)
        if save_numeric_data:
            # save on neptune
            neptune_run[f"Unseen / Query / {model_name} / Class {k} data"].upload(File.as_pickle(score))
            # save file locally
            filepath = path + f'/Class_{k}_data'
            with open(filepath, 'wb') as f:
                pickle.dump(score, f)


def getCheckpointDir(params, configs, suffix):
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
    checkpoint_dir = checkpoint_dir + "_" + suffix

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

    models = {}

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
    model = model.cuda()

    suffix = 'fhyper_maml_exp'
    params_experiment.method = 'fhyper_maml'
    checkpoint_dir = getCheckpointDir(params_experiment, configs, suffix)

    modelfile = get_best_file(checkpoint_dir)  # load best from given model
    print("Using model file", modelfile)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
    else:
        print("[WARNING] Cannot find 'best_file.tar' in: " + str(params_experiment.checkpoint_dir))

    models['fhyper_maml'] = model
    #######################################
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
    model = model.cuda()

    suffix = 'bayes_hmaml_exp'
    params_experiment.method = 'bayes_hmaml'
    params_experiment.checkpoint_dir = getCheckpointDir(params_experiment, configs, suffix)

    modelfile = get_best_file(params_experiment.checkpoint_dir)  # load best from given model
    print("Using model file", modelfile)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
    else:
        print("[WARNING] Cannot find 'best_file.tar' in: " + str(params_experiment.checkpoint_dir))

    models['bayes_hmaml'] = model        

    neptune_run = setup_neptune(params_experiment)
    #neptune_run = None
    # primary batches for adaptation

    x, labels1 = next(iter(val_loader))
    features = x

    for _, model in models.items():
        model.n_query = features.size(1) - model.n_support
        model.train()

    data = features
    features = features.cuda()
    x_var = torch.autograd.Variable(features)
    support_data1 = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                            *x.size()[2:])  # support data
    query_data1 = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                        *x.size()[2:])  # query data

    # new batch for experiment
    features2, labels2 = next(iter(val_loader))
    # print('finding val batch')
    # if there are repetitions between batches get another batch
    while reduce(np.intersect1d, (labels1, labels2)).size > 0:
        features2, labels2 = next(iter(val_loader))
    print(labels2)
    features_unseen = features2

    for _, model in models.items():
        model.n_query = features2.size(1) - model.n_support
    # no tuning
    features_unseen = features_unseen.cuda()
    x2_var = torch.autograd.Variable(features_unseen)
    support_data2 = x2_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                            *features2.size()[2:])  # support data
    query_data2 = x2_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                            *features2.size()[2:])  # query data
    for _, model in models.items():
        model.eval()
        # model.train()

    for model_name, model in models.items():
        print("Model name:", model_name)
        s1 = []
        q1 = []
        s2 = []
        q2 = []

        calibration_error = np.zeros(4, float)
        
        for num in range(num_samples):
            for k,weight in enumerate(model.classifier.parameters()):
                weight.fast=None
            if model_name == 'fhyper_maml':
                model.manager.clear_all_fields()
            model.set_forward(data)

            out = F.softmax(model(support_data1), dim=1)
            s1.append(out[0].clone().data.cpu().numpy())
            s1_labels = torch.from_numpy(np.repeat(range(model.n_way), model.n_support)).cuda()
            calibration_error[0] += MCE(out, s1_labels, num_classes = 5, validate_args = False).item()
            
            out = F.softmax(model(query_data1), dim=1)
            q1.append(out[0].clone().data.cpu().numpy())
            q1_labels = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
            calibration_error[1] += MCE(out, q1_labels, num_classes = 5, validate_args = False).item()
            if num % 25 == 0:
                print(f'Accuracy on query at: {num}/{num_samples}')
                query_data_labels = torch.from_numpy(np.repeat(range(n_way), n_query))
                topk_scores, topk_labels = model(query_data1).data.topk(1, 1, True, True)
                topk_ind = topk_labels.cpu().numpy().flatten()
                y_labels = query_data_labels.cpu().numpy()
                top1_correct = np.sum(topk_ind == y_labels)
                task_accuracy = (top1_correct / len(query_data_labels)) * 100
                print(task_accuracy)

            out = F.softmax(model(support_data2), dim=1)
            s2.append(out[0].clone().data.cpu().numpy())
            s2_labels = torch.from_numpy(np.repeat(range(model.n_way), model.n_support)).cuda()
            calibration_error[2] += MCE(out, s2_labels, num_classes = 5, validate_args = False).item()

            out = F.softmax(model(query_data2), dim=1)
            q2.append(out[0].clone().data.cpu().numpy())
            q2_labels = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
            calibration_error[3] += MCE(out, q2_labels, num_classes = 5, validate_args = False).item()
            if num % 25 == 0:
                query_data_labels = torch.from_numpy(np.repeat(range(n_way), n_query))
                topk_scores, topk_labels = model(query_data2).data.topk(1, 1, True, True)
                topk_ind = topk_labels.cpu().numpy().flatten()
                y_labels = query_data_labels.cpu().numpy()
                top1_correct = np.sum(topk_ind == y_labels)
                task_accuracy = (top1_correct / len(query_data_labels)) * 100
                print(task_accuracy)

        neptune_run[f'{model_name}/s1'] = calibration_error[0] / num_samples
        neptune_run[f'{model_name}/q1'] = calibration_error[1] / num_samples
        neptune_run[f'{model_name}/s2'] = calibration_error[2] / num_samples
        neptune_run[f'{model_name}/q2'] = calibration_error[3] / num_samples
        plot_histograms(neptune_run, model_name, s1, s2, q1, q2)


def main():
    # params_experiment = parse_args('train')
    params_experiment = parse_args('experiment1')   # todo sprawdzic parametry i wywalic najlepiej
    experiment(params_experiment=params_experiment)


if __name__ == '__main__':
    main()
