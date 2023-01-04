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



