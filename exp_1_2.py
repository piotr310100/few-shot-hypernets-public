import pickle
import shutil
from pathlib import Path
from functools import reduce
import torch
import torch.optim
import torch.utils.data.sampler
from torch.nn import functional as F
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
from methods.hypernets.hypermaml import HyperMAML



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


def initLocalDirectories(methods):
    if path.isdir('exp_1_data'):
        shutil.rmtree('exp_1_data')
    os.mkdir('exp_1_data')



def experiment(params_experiment):

    initLocalDirectories(['hyper_maml', 'fhyper_maml', 'bayes_hmaml'])
    num_samples = params_experiment.num_samples
    support_file = configs.data_dir['omniglot'] + 'noLatin.json'
    query_file = configs.data_dir['emnist'] + 'val.json'

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
    base_datamgr = SetDataManager(image_size, **train_few_shot_params)  # n_eposide = 100
    base_loader = base_datamgr.get_data_loader(support_file, aug=params_experiment.train_aug)

    test_few_shot_params = dict(n_way=n_way, n_support=params_experiment.n_shot, n_query=n_query)
    val_datamgr = SetDataManager(image_size, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(query_file, aug=False)

    if params_experiment.dataset in ['omniglot', 'cross_char']:
        assert params_experiment.model == 'Conv4' and not params_experiment.train_aug, 'omniglot only support Conv4 without augmentation'

    models = {}

    backbone.ConvBlock.maml = True
    backbone.SimpleBlock.maml = True
    backbone.BottleneckBlock.maml = True
    backbone.ResNet.maml = True
    model = HyperMAML(model_dict[params_experiment.model], params=params_experiment,
                       approx=(params_experiment.method == 'maml_approx'),
                       **train_few_shot_params)

    if params_experiment.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
        model.n_task = 32
        model.train_lr = 0.1
    model = model.cuda()

    suffix = 'hyper_maml_exp'
    params_experiment.method = 'hyper_maml'
    # checkpoint_dir = getCheckpointDir(params_experiment, configs, suffix)
    # add your own filepath
    checkpoint_dir = './save/checkpoints/cross_char/Conv4_hyper_maml_5way_1shot_fhmaml_best'
    modelfile = get_best_file(checkpoint_dir)  # load best from given model
    print("Using model file", modelfile)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
    else:
        print("[WARNING] Cannot find 'best_file.tar' in: " + str(params_experiment.checkpoint_dir))

    models['hyper_maml'] = model
    #######################################
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
    # checkpoint_dir = getCheckpointDir(params_experiment, configs, suffix)
    # add your own filepath
    checkpoint_dir = './save/checkpoints/cross_char/Conv4_fhyper_maml_5way_1shot_optionAtomic_0_flow_w_1e-6'
    # .\save\checkpoints\cross_char\Conv4_fhyper_maml_5way_1shot_optionAtomic_0_flow_w_1e-6
    # .\save\checkpoints\cross_char\Conv4_hyper_maml_5way_1shot_3_512_crosschar_lr0.01_wsn5_val0.001
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
    model.weight_set_num_test = 1
    model.weight_set_num_train = 1

    if params_experiment.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
        model.n_task = 32
        model.train_lr = 0.1   
    model = model.cuda()

    suffix = 'bayes_hmaml_exp'
    params_experiment.method = 'bayes_hmaml'
    # add your own filepath
    #params_experiment.checkpoint_dir = getCheckpointDir(params_experiment, configs, suffix)
    params_experiment.checkpoint_dir = './save/checkpoints/cross_char/Conv4_hyper_maml_5way_1shot_3_512_crosschar_lr0.01_wsn5_val0.001'

    modelfile = get_best_file(params_experiment.checkpoint_dir)  # load best from given model
    print("Using model file", modelfile)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
    else:
        print("[WARNING] Cannot find 'best_file.tar' in: " + str(params_experiment.checkpoint_dir))

    models['bayes_hmaml'] = model        

    #neptune_run = setup_neptune(params_experiment)
    #neptune_run = None
    # primary batches for adaptation

    features1, labels1 = next(iter(base_loader))


    for _, model in models.items():
        model.n_query = features1.size(1) - model.n_support
        model.train()

    data = features1
    features1 = features1.cuda()
    x_var = torch.autograd.Variable(features1)
    support_data1 = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                            *features1.size()[2:])  # support data
    query_data1 = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                        *features1.size()[2:])  # query data

    # new batch for experiment
    features2, labels2 = next(iter(val_loader))
    # # print('finding val batch')
    # # if there are repetitions between batches get another batch
    # while reduce(np.intersect1d, (labels1, labels2)).size > 0:
    #     features2, labels2 = next(iter(val_loader))
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
        
        for num in range(num_samples):
            for k, weight in enumerate(model.classifier.parameters()):
                weight.fast=None
            if model_name == 'fhyper_maml':
                model.manager.clear_all_fields()

            model.set_forward(data)

            outs1 = F.softmax(model(support_data1), dim=1)
            outq1 = F.softmax(model(query_data1), dim=1)
            outs2 = F.softmax(model(support_data2), dim=1)
            outq2 = F.softmax(model(query_data2), dim=1)

            s1.append(outs1.clone().data.cpu().numpy()) # czyli tutaj bedzie 1000 list po 5 elementow
            q1.append(outq1.clone().data.cpu().numpy())
            s2.append(outs2.clone().data.cpu().numpy())
            q2.append(outq2.clone().data.cpu().numpy())

            if num % 25 == 0:
                print(f'Accuracy on query at: {num}/{num_samples}')
                query_data_labels = torch.from_numpy(np.repeat(range(n_way), n_query))
                topk_scores, topk_labels = model(query_data1).data.topk(1, 1, True, True)
                topk_ind = topk_labels.cpu().numpy().flatten()
                y_labels = query_data_labels.cpu().numpy()
                top1_correct = np.sum(topk_ind == y_labels)
                task_accuracy = (top1_correct / len(query_data_labels)) * 100
                print(task_accuracy)
                topk_scores, topk_labels = model(query_data2).data.topk(1, 1, True, True)
                topk_ind = topk_labels.cpu().numpy().flatten()
                top1_correct = np.sum(topk_ind == y_labels)
                task_accuracy = (top1_correct / len(query_data_labels)) * 100
                print(task_accuracy)

        s1 = np.array(s1)   # 1000 x 5 x 5 dla 5 taskow
        s2 = np.array(s2)   # 1000 x 5 x 5 dla 5 taskow
        q1 = np.array(q1)
        q2 = np.array(q2)
        curpath = os.path.abspath(os.curdir)

        with(open(f'./exp_1_data/{model_name}_fullUnseenSupport_DATA', 'wb')) as f:
            pickle.dump(s2, f)
        with(open(f'./exp_1_data/{model_name}_fullSeenSupport_DATA', 'wb')) as f:
            pickle.dump(s1, f)
        with(open(f'./exp_1_data/{model_name}_fullUnseenQuery_DATA', 'wb')) as f:
            pickle.dump(q2, f)
        with(open(f'./exp_1_data/{model_name}_fullSeenQuery_DATA', 'wb')) as f:
            pickle.dump(q1, f)

def dump_file(file_name,file):
    curpath = os.path.abspath(os.curdir)
    with(open(os.path.join(curpath,'exp_1_data',file_name)), 'wb') as f:
        pickle.dump(file,f)

def main():
    # params_experiment = parse_args('train')
    params_experiment = parse_args('experiment1')
    experiment(params_experiment=params_experiment)


if __name__ == '__main__':
    main()
