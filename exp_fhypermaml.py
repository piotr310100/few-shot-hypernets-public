<<<<<<< Updated upstream
from pathlib import Path

import torch
import torch.optim
import torch.utils.data.sampler
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from neptune.new.types import File
from torch.autograd import Variable
import configs
from data.datamgr import SetDataManager

from methods.hypernets.fhypermaml import FHyperMAML
from io_utils import model_dict, parse_args, get_best_file , setup_neptune

params = parse_args('train')

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

n_query = max(1,
              int(16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot, n_query=n_query)
base_datamgr = SetDataManager(image_size, **train_few_shot_params)  # n_eposide=100
base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=n_query)
val_datamgr = SetDataManager(image_size, **test_few_shot_params)
val_loader = val_datamgr.get_data_loader(val_file, aug=False)

if params.dataset in ['omniglot', 'cross_char']:
    assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
    # params.model = 'Conv4S'

if params.method == 'hyper_maml':
    model = FHyperMAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'),
                      **train_few_shot_params)
    if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
        model.n_task = 32
        model.train_lr = 0.1
else:
    raise ValueError('Unknown method')

model = model.cuda()

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

# modelfile   = get_resume_file(checkpoint_dir)

params.checkpoint_dir = checkpoint_dir

if not params.method in ['baseline', 'baseline++']:
    modelfile = get_best_file(checkpoint_dir)

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
# metrics = model.set_forward_loss(x)
is_feature=False
x = x.cuda()
x_var = Variable(x)
support_data = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                    *x.size()[2:])  # support data
query_data = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                  *x.size()[2:])  # query data
support_data_labels = model.get_support_data_labels()
support_embeddings = model.feature(support_data)
if model.hm_detach_feature_net:
    support_embeddings = support_embeddings.detach()

if model.hm_detach_before_hyper_net:
    support_embeddings = support_embeddings.detach()

delta_params_list = []
total_loss_flow = None

model.num_points_train = params.num_points_train
model.flow.epoch_property.temp_w = 1
model.flow.epoch_property.dkl_w = 1

train_stage = True
norm_warmup = False
weights_delta = None
bias_delta = None
for name, param_net in model.hypernet_heads.items():
    support_embeddings_resh = support_embeddings.reshape(
        model.n_way, -1
    )
    delta_params = param_net(support_embeddings_resh)
    bias_neurons_num = model.target_net_param_shapes[name][0] // model.n_way
    if model.hn_adaptation_strategy == 'increasing_alpha' and model.alpha < 1:
        delta_params = delta_params * model.alpha
    delta_params_shape = delta_params.shape

    delta_params, loss_flow = model.flow(delta_params, train_stage, norm_warmup)
    density_loss = model.flow.get_density_loss(delta_params)
    loss_flow = model.flow.epoch_property.dkl_w * (loss_flow - density_loss)

    if total_loss_flow is None:
        total_loss_flow = loss_flow
    else:
        total_loss_flow = total_loss_flow + loss_flow
    delta_params = delta_params.reshape(-1, *delta_params_shape)
    weights_delta = delta_params[:, :, :-bias_neurons_num].detach().cpu()
    bias_delta = delta_params[:, :, -bias_neurons_num:].squeeze().detach().cpu()
    delta_params_list.extend([weights_delta, bias_delta])

print(delta_params_list)
torch.save(weights_delta, 'weights_delta.pt')
torch.save(bias_delta, 'bias_delta.pt')
=======
from typing import Type, List, Union, Dict, Optional

import torch.optim
import torch.utils.data.sampler
from torch.autograd import Variable
import numpy as np
import torch
import random
from neptune.new import Run
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import os
from torch.nn import functional as F
import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.DKT import DKT
from methods.hypernets.hypernet_poc import HyperNetPOC
from methods.hypernets import hypernet_types
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.hypernets.bayeshmaml import BayesHMAML
from methods.hypernets.hypermaml import HyperMAML
from methods.hypernets.fhypermaml import FHyperMAML
from io_utils import model_dict, parse_args, get_resume_file, setup_neptune


def _set_seed(seed, verbose=True):
    if (seed != 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if (verbose): print("[INFO] Setting SEED: " + str(seed))
    else:
        if (verbose): print("[INFO] Setting SEED: None")


params = parse_args('train')
if __name__ == '__main__':
    _set_seed(params.seed)
    base_file = configs.data_dir['omniglot'] + 'noLatin.json'
    val_file = configs.data_dir['emnist'] + 'val.json'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        # params.model = 'Conv4S'
        # no need for this, since omniglot is loaded as RGB

    # optimization = 'Adam'
    optimization = params.optim

    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600  # default

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=16)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(model_dict[params.model], params.num_classes, loss_type='dist')

    elif params.method in ['DKT', 'protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml',
                           'maml_approx', 'hyper_maml', 'bayes_hmaml', 'fhyper_maml'] + list(hypernet_types.keys()):
        n_query = max(1, int(
            16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        print("n_query", n_query)
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot, n_query=n_query)
        base_datamgr = SetDataManager(image_size, **train_few_shot_params)  # n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=n_query)

        val_datamgr = SetDataManager(image_size, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if (params.method == 'DKT'):
            dkt_train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
            model = DKT(model_dict[params.model], **dkt_train_few_shot_params)
            model.init_summary()
        elif params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'),
                         **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1

        elif params.method in hypernet_types.keys():
            hn_type: Type[HyperNetPOC] = hypernet_types[params.method]
            model = hn_type(model_dict[params.model], params=params, **train_few_shot_params)
        elif params.method in ['fhyper_maml', 'hyper_maml', 'bayes_hmaml']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            if params.method == 'bayes_hmaml':
                model = BayesHMAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'),
                                   **train_few_shot_params)
            elif params.method == 'hyper_maml':
                model = HyperMAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'),
                                  **train_few_shot_params)
            else:
                model = FHyperMAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'),
                                   **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)

    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    if params.checkpoint_suffix != "":
        params.checkpoint_dir = params.checkpoint_dir + "_" + params.checkpoint_suffix
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params.checkpoint_dir)
    start_epoch = params.start_epoch

    stop_epoch = params.stop_epoch * model.n_task + 1  # maml use multiple tasks in one update

    resume_file = get_resume_file(params.checkpoint_dir)
    print(resume_file)
    if resume_file is not None:
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch'] + 1
        model.load_state_dict(tmp['state'])
        print("Resuming training from", resume_file, "epoch", start_epoch)

    args_dict = vars(params.params)
    # with (Path(params.checkpoint_dir) / "args.json").open("w") as f:
    #     json.dump(
    #         {
    #             k: v if isinstance(v, (int, str, bool, float)) else str(v)
    #             for (k, v) in args_dict.items()
    #         },
    #         f,
    #         indent=2,
    #     )
    model.train()

    x, out1 = next(iter(val_loader))
    model.n_query = x.size(1) - model.n_support
    # metrics = model.set_forward_loss(x)
    is_feature = False
    x = x.cuda()
    x_var = Variable(x)
    support_data = x_var[:, :model.n_support, :, :, :].contiguous().view(model.n_way * model.n_support,
                                                                         *x.size()[2:])  # support data
    query_data = x_var[:, model.n_support:, :, :, :].contiguous().view(model.n_way * model.n_query,
                                                                       *x.size()[2:])  # query data
    support_data_labels = model.get_support_data_labels()
    support_embeddings = model.feature(support_data)
    if model.hm_detach_feature_net:
        support_embeddings = support_embeddings.detach()

    if model.hm_detach_before_hyper_net:
        support_embeddings = support_embeddings.detach()

    with torch.no_grad():
        logits = model.classifier.forward(support_embeddings).detach()
        logits = F.softmax(logits, dim=1)

    labels = support_data_labels.view(support_embeddings.shape[0], -1)
    support_embeddings = torch.cat((support_embeddings, logits, labels), dim=1)

    for weight in model.parameters():
        weight.fast = None
    model.zero_grad()

    support_embeddings = model.apply_embeddings_strategy(support_embeddings)
    delta_params_list = []
    total_loss_flow = None

    model.num_points_train = params.num_points_train
    model.flow.epoch_property.temp_w = 1
    model.flow.epoch_property.dkl_w = 1
    weights_delta_all = []
    bias_delta_all = []
    delta_params_all = []
    train_stage = True
    norm_warmup = False
    weights_delta = None
    bias_delta = None
    num_samples = 1000

    model.flow.num_points_train = model.flow.num_points_test = 1
    for num_sample in range(num_samples):
        print(num_sample)
        for name, param_net in model.hypernet_heads.items():
            support_embeddings_resh = support_embeddings.reshape(
                model.n_way, -1
            )
            delta_params = param_net(support_embeddings_resh)
            bias_neurons_num = model.target_net_param_shapes[name][0] // model.n_way
            if model.hn_adaptation_strategy == 'increasing_alpha' and model.alpha < 1:
                delta_params = delta_params * model.alpha
            delta_params_shape = delta_params.shape

            delta_params, loss_flow = model.flow(delta_params, train_stage, norm_warmup)
            density_loss = model.flow.get_density_loss(delta_params)
            loss_flow = model.flow.epoch_property.dkl_w * (loss_flow - density_loss)

            if total_loss_flow is None:
                total_loss_flow = loss_flow
            else:
                total_loss_flow = total_loss_flow + loss_flow
            delta_params = delta_params.reshape(-1, *delta_params_shape)
            weights_delta = delta_params[:, :, :-bias_neurons_num].detach().cpu()
            bias_delta = delta_params[:, :, -bias_neurons_num:].squeeze().detach().cpu()
            delta_params_list.extend([weights_delta, bias_delta])
            #print(delta_params_list)

            delta_params_all.append(delta_params.flatten().detach().cpu())
            bias_delta_all.append(bias_delta.flatten())
            weights_delta_all.append(weights_delta.flatten())

    torch.save(weights_delta_all, 'weights_delta_all.pt')
    torch.save(bias_delta_all, 'bias_delta_all.pt')
    torch.save(delta_params_all, 'delta_params_all.pt')
>>>>>>> Stashed changes


