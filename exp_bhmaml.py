import torch.optim
import torch.utils.data.sampler
from torch.autograd import Variable
import numpy as np
import torch
import random
import torch.optim
import os
from torch.nn import functional as F
import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.hypernets import hypernet_types
from methods.hypernets.bayeshmaml import BayesHMAML
from io_utils import model_dict, parse_args, get_resume_file


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

        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = BayesHMAML(model_dict[params.model], params=params, approx=(params.method == 'maml_approx'),
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

    weights_delta_mean_all = []
    bias_delta_mean_all = []
    weights_logvar_all = []
    bias_logvar_all = []
    num_samples = 1000
    # model.weight_set_num_train = model.hm_weight_set_num_test = 1

    for num_sample in range(num_samples):
        print(num_sample)
        for name, param_net in model.hypernet_heads.items():

            support_embeddings_resh = support_embeddings.reshape(
                model.n_way, -1
            )

            delta_params_mean, params_logvar = param_net(support_embeddings_resh)
            bias_neurons_num = model.target_net_param_shapes[name][0] // model.n_way

            if model.hn_adaptation_strategy == 'increasing_alpha' and model.alpha < 1:
                delta_params_mean = delta_params_mean * model.alpha
                params_logvar = params_logvar * model.alpha

            weights_delta_mean = delta_params_mean[:, :-bias_neurons_num].contiguous().view(
                *model.target_net_param_shapes[name])
            bias_delta_mean = delta_params_mean[:, -bias_neurons_num:].flatten()

            weights_logvar = params_logvar[:, :-bias_neurons_num].contiguous().view(
                *model.target_net_param_shapes[name])
            bias_logvar = params_logvar[:, -bias_neurons_num:].flatten()

            delta_params_list.append([weights_delta_mean, weights_logvar])
            delta_params_list.append([bias_delta_mean, bias_logvar])

            weights_delta_mean_all.append(weights_delta_mean)
            bias_delta_mean_all.append(bias_delta_mean)
            weights_logvar_all.append(weights_logvar)
            bias_logvar_all.append(bias_logvar)

    torch.save(weights_delta_mean_all, 'exp_bhmaml_evaluation/weights_delta_mean_all.pt')
    torch.save(bias_delta_mean_all, 'exp_bhmaml_evaluation/bias_delta_mean_all.pt')
    torch.save(weights_logvar_all, 'exp_bhmaml_evaluation/weights_logvar_all.pt')
    torch.save(bias_logvar_all, 'exp_bhmaml_evaluation/bias_logvar_all.pt')