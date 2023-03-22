from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from time import time

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import backbone
from methods.hypernets.utils import get_param_dict, accuracy_from_scores
from methods.maml import MAML
from regressionFlow.models.networks_regression_SDD_conditional import CRegression
from methods.hypernets.hypermaml import HyperNet

# FlowHyperMAML (HyperMAML with modified loss calculated with regFlow)
class FHyperMAML(MAML):
    class _MetricsManager:
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
            self.delta_theta_std = []
            self.flow_w = flow_w
            self.atribs = ['acc', 'loss', 'loss_ce', 'flow_loss','flow_loss_scaled', 'flow_density_loss','flow_loss_raw',
                           'theta_norm', 'delta_theta_norm','delta_theta_std']

        def clear_field(self,atrib_name):
            self.assert_exist(atrib_name)
            getattr(self,atrib_name).clear()
        def clear_all_fields(self):
            for atrib in self.atribs:
                getattr(self, atrib).clear()
        def get_metrics(self,clean_after:bool=True):
            for atrib in self.atribs:
                if not getattr(self,atrib):
                    if atrib == 'loss':
                        self.append(atrib, torch.tensor([0]).cuda())
                    else:
                        self.append(atrib, 0)
            #print(self.flow_loss)
            out = {'accuracy/train': np.asarray(self.acc).mean(),
                    'loss': torch.stack(self.loss).mean(dtype=torch.float).item(),  # loss := loss_ce - flow_w * loss_flow
                    'loss_ce':np.asarray(self.loss_ce).mean(),
                    'flow_loss': np.asarray(self.flow_loss).mean(),    # loss_flow := flow_output_loss - density_loss (before scaling with flow_w)
                    'flow_loss_scaled': np.asarray(self.flow_loss_scaled).mean(),  # loss_flow * flow_w
                    'flow_density_loss': np.asarray(self.flow_density_loss).mean(),    # density component before scaling with flow_w
                    'flow_loss_raw': np.asarray(self.flow_loss_raw).mean(), # loss_flow before substracting density component (and before scaling with flow_w)
                    'theta_norm': np.asarray(self.theta_norm).mean(),
                    'delta_theta_norm': np.asarray(self.delta_theta_norm).mean(),
                    'delta_theta_std': np.asarray(self.delta_theta_std).mean()
                    }
            if clean_after:
                self.clear_all_fields()

            return out
        def append(self, atrib_name, value):
            self.assert_exist(atrib_name)
            if type(value) is torch.Tensor:
                value = value.squeeze().cuda()
            if atrib_name == 'flow_loss' and self.flow_w is not None:
                self.flow_loss_scaled.append(value * self.flow_w)
            getattr(self, atrib_name).append(value)

    def __init__(self, model_func, n_way, n_support, n_query, params=None, approx=False):
        super(FHyperMAML, self).__init__(model_func, n_way, n_support, n_query, params=params)
        self.loss_fn = nn.CrossEntropyLoss()
        self.hn_tn_hidden_size = params.hn_tn_hidden_size
        self.hn_tn_depth = params.hn_tn_depth
        self._init_classifier()

        self.enhance_embeddings = params.hm_enhance_embeddings

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

        self.hn_sup_aggregation = params.hn_sup_aggregation
        self.hn_hidden_size = params.hn_hidden_size
        self.hm_lambda = params.hm_lambda
        self.hm_save_delta_params = params.hm_save_delta_params
        self.hm_use_class_batch_input = params.hm_use_class_batch_input
        self.hn_adaptation_strategy = params.hn_adaptation_strategy
        self.hm_support_set_loss = params.hm_support_set_loss
        self.hm_maml_warmup = params.hm_maml_warmup
        self.hm_maml_warmup_coef = 0
        self.hm_maml_warmup_epochs = params.hm_maml_warmup_epochs
        self.hm_maml_warmup_switch_epochs = params.hm_maml_warmup_switch_epochs
        self.hm_maml_update_feature_net = params.hm_maml_update_feature_net
        self.hm_update_operator = params.hm_update_operator
        self.hm_load_feature_net = params.hm_load_feature_net
        self.hm_feature_net_path = params.hm_feature_net_path
        self.hm_detach_feature_net = params.hm_detach_feature_net
        self.hm_detach_before_hyper_net = params.hm_detach_before_hyper_net
        self.hm_set_forward_with_adaptation = params.hm_set_forward_with_adaptation
        self.hn_val_lr = params.hn_val_lr
        self.hn_val_epochs = params.hn_val_epochs
        self.hn_val_optim = params.hn_val_optim

        self.alpha = 0
        self.hn_alpha_step = params.hn_alpha_step

        if self.hn_adaptation_strategy == 'increasing_alpha' and self.hn_alpha_step < 0:
            raise ValueError('hn_alpha_step is not positive!')

        self.single_test = False
        self.epoch = -1
        self.start_epoch = -1
        self.stop_epoch = -1

        self.calculate_embedding_size()

        self._init_hypernet_modules(params)
        self._init_feature_net()

        self.flow_num_temperature_warmup_epochs = params.flow_temp_warmup_epochs
        self.flow_temp_strategy = params.flow_temp_strategy
        self.num_points_train = params.num_points_train
        self.num_points_test = params.num_points_test

        self.dkl_downfall_strategy = params.dkl_downfall_strategy
        self.dkl_downfall_magnitude = params.dkl_downfall_magnitude
        self.dkl_downfall_stop_epoch = params.dkl_downfall_stop_epoch
        self.dkl_downfall_start_epoch = params.dkl_downfall_start_epoch
        self.dkl_downfall_linear_delta = None

        self.flow_args = Namespace(shape1=5, shape2=65,
                                   model_type='PointNet', logprob_type='Normal', input_dim=325, dims='500',
                                   latent_dims='256', hyper_dims='256', num_blocks=1, latent_num_blocks=1,
                                   layer_type='concatsquash', time_length=0.5, train_T=True, nonlinearity='tanh',
                                   use_adjoint=True, solver='dopri5', atol=1e-05, rtol=1e-05, batch_norm=False,
                                   sync_bn=False, bn_lag=0, zdim=params.flow_zdim, num_points_train = self.num_points_train,
                                   num_points_test = self.num_points_test,
                                   #    ------  DO TEGO MIEJSCA SA WAZNE ARGUMENTY ARCHITEKTURY flowa   ---------
                                   root_dir=None, use_latent_flow=False,
                                   use_deterministic_encoder=False,
                                   optimizer='adam', batch_size=1000, lr=0.001, beta1=0.9,
                                   beta2=0.999, momentum=0.9, weight_decay=1e-05, epochs=1000, seed=694754,
                                   recon_weight=1.0, prior_weight=1.0, entropy_weight=1.0, scheduler='linear',
                                   exp_decay=1.0, exp_decay_freq=1, image_size='28x28', data_dir='data/SDD/',
                                   dataset_type='shapenet15k', cates=['airplane'],
                                   mn40_data_dir='data/ModelNet40.PC15k',
                                   mn10_data_dir='data/ModelNet10.PC15k', dataset_scale=1.0, random_rotate=False,
                                   normalize_per_shape=False, normalize_std_per_axis=False, tr_max_sample_points=2048,
                                   te_max_sample_points=2048, num_workers=4, use_all_data=False,
                                   log_name='experiment_regression_flow_toy', viz_freq=1, val_freq=10, log_freq=1,
                                   save_freq=5, no_validation=False, save_val_results=False, eval_classification=False,
                                   no_eval_sampling=False, max_validate_shapes=None, resume_checkpoint=None,
                                   resume_optimizer=False, resume_non_strict=False, resume_dataset_mean=None,
                                   resume_dataset_std=None, world_size=1, dist_url='tcp://127.0.0.1:9991',
                                   dist_backend='nccl', distributed=False, rank=0, gpu=0, evaluate_recon=False,
                                   num_sample_shapes=10, num_sample_points=2048, use_sphere_dist=False,
                                   use_div_approx_train=False, use_div_approx_test=False)

        self.flow = CRegression(self.flow_args)

        # args for scaling flow temp and dkl
        self.flow_w = params.flow_w
        self.manager = self._MetricsManager(self.flow_w)
        self.flow_temp_scale = params.flow_temp_scale
        self.flow_temp_stop_val = params.flow_temp_stop_val
        self.flow_temp_step = None

    def _sample_temp_step(self):
        if self.dkl_downfall_strategy == 'None':
            self.flow.epoch_property.temp_w = self.flow_temp_stop_val
            return
        if self.flow_temp_strategy == "Exp":
            if self.flow_num_temperature_warmup_epochs > self.flow.epoch_property.curr_epoch and self.flow_num_temperature_warmup_epochs > 0:
                if self.flow_temp_step is None:
                    self.flow.epoch_property.temp_w = 1
                    self.flow_temp_step = np.power(1 / self.flow_temp_scale * self.flow_temp_stop_val,
                                                   1 / self.flow_num_temperature_warmup_epochs)
                self.flow_temp_scale = self.flow_temp_scale * self.flow_temp_step          
        elif self.flow_temp_strategy == "Linear":
            if self.flow_num_temperature_warmup_epochs > self.flow.epoch_property.curr_epoch and self.flow_num_temperature_warmup_epochs > 0:
                if self.flow_temp_step is None:
                    self.flow.epoch_property.temp_w = 0
                    self.flow_temp_step = self.flow_temp_stop_val / self.flow_num_temperature_warmup_epochs
                self.flow_temp_scale = self.flow_temp_scale + self.flow_temp_step

    def _update_flow(self):
        assert self.flow_temp_scale > 0
        
        if self.single_test:
            self.flow.epoch_property.temp_w = self.flow_temp_stop_val
            return
        
        if self.flow_num_temperature_warmup_epochs <= self.flow.epoch_property.curr_epoch \
                or self.flow.epoch_property.temp_w > self.flow_temp_stop_val:  # any numeric errors
            self.flow.epoch_property.temp_w = self.flow_temp_stop_val
        else:
            self.flow.epoch_property.temp_w = self.flow_temp_scale

    def _init_feature_net(self):
        if self.hm_load_feature_net:
            print(f'loading feature net model from location: {self.hm_feature_net_path}')
            model_dict = torch.load(self.hm_feature_net_path)
            self.feature.load_state_dict(model_dict['state'])

    def _init_classifier(self):
        assert self.hn_tn_hidden_size % self.n_way == 0, f"hn_tn_hidden_size {self.hn_tn_hidden_size} should be the multiple of n_way {self.n_way}"
        layers = []

        for i in range(self.hn_tn_depth):
            in_dim = self.feat_dim if i == 0 else self.hn_tn_hidden_size
            out_dim = self.n_way if i == (self.hn_tn_depth - 1) else self.hn_tn_hidden_size

            linear = backbone.FLinear_fw(in_dim, out_dim)
            linear.bias.data.fill_(0)

            layers.append(linear)

        self.classifier = nn.Sequential(*layers)

    def _init_hypernet_modules(self, params):
        target_net_param_dict = get_param_dict(self.classifier)

        target_net_param_dict = {
            name.replace(".", "-"): p
            # replace dots with hyphens bc torch doesn't like dots in modules names
            for name, p in target_net_param_dict.items()
        }

        self.target_net_param_shapes = {
            name: p.shape
            for (name, p)
            in target_net_param_dict.items()
        }

        self.hypernet_heads = nn.ModuleDict()

        for name, param in target_net_param_dict.items():
            if self.hm_use_class_batch_input and name[-4:] == 'bias':
                continue

            bias_size = param.shape[0] // self.n_way

            head_in = self.embedding_size
            head_out = (param.numel() // self.n_way) + bias_size if self.hm_use_class_batch_input else param.numel()

            self.hypernet_heads[name] = HyperNet(self.hn_hidden_size, self.n_way, head_in, self.feat_dim, head_out,
                                                 params)

    def calculate_embedding_size(self):
        n_classes_in_embedding = 1 if self.hm_use_class_batch_input else self.n_way
        n_support_per_class = 1 if self.hn_sup_aggregation == 'mean' else self.n_support
        single_support_embedding_len = self.feat_dim + self.n_way + 1 if self.enhance_embeddings else self.feat_dim
        self.embedding_size = n_classes_in_embedding * n_support_per_class * single_support_embedding_len

    def apply_embeddings_strategy(self, embeddings):
        if self.hn_sup_aggregation == 'mean':
            new_embeddings = torch.zeros(self.n_way, *embeddings.shape[1:])

            for i in range(self.n_way):
                lower = i * self.n_support
                upper = (i + 1) * self.n_support
                new_embeddings[i] = embeddings[lower:upper, :].mean(dim=0)

            return new_embeddings.cuda()

        return embeddings

    def get_support_data_labels(self):
        return torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()  # labels for support data

    def get_hn_delta_params(self, support_embeddings, train_stage):
        if self.hm_detach_before_hyper_net:
            support_embeddings = support_embeddings.detach()

        if self.hm_use_class_batch_input:
            delta_params_list = []
            density_loss_list = []
            raw_loss_flow_list = []
            delta_params_std_list = []
            total_loss_flow = None
            for name, param_net in self.hypernet_heads.items():

                support_embeddings_resh = support_embeddings.reshape(
                    self.n_way, -1
                )

                delta_params = param_net(support_embeddings_resh)
                bias_neurons_num = self.target_net_param_shapes[name][0] // self.n_way
                if self.hn_adaptation_strategy == 'increasing_alpha' and self.alpha < 1:
                    delta_params = delta_params * self.alpha
                delta_params_shape = delta_params.shape
                if self.hm_maml_warmup:
                    self._update_hm_maml_warmup_coef()

                flow_pass = self.hm_maml_warmup_coef < 1 or self.single_test
                if flow_pass:
                    self._update_flow()
                    if self.single_test:
                        cond = False
                    else:
                        cond = train_stage
                    delta_params, loss_flow = self.flow(delta_params, cond)
                    delta_params_std_list.append(torch.mean(torch.std(delta_params, dim=1)).item())
                    density_loss = self.flow.get_density_loss(delta_params)
                    raw_loss_flow_list.append(loss_flow)
                    density_loss_list.append(density_loss)  # before scaling
                    loss_flow = self.flow.epoch_property.dkl_w * (loss_flow - density_loss)
                else:
                    loss_flow = torch.tensor([0])

                if total_loss_flow is None:
                    total_loss_flow = loss_flow
                else:
                    total_loss_flow = total_loss_flow + loss_flow


                self.manager.append('delta_theta_norm', torch.linalg.vector_norm(delta_params, dim=1).mean().item())
                delta_params = delta_params.reshape(-1, *delta_params_shape)
                weights_delta = delta_params[:, :, :-bias_neurons_num]
                bias_delta = delta_params[:, :, -bias_neurons_num:].squeeze(-1)
                delta_params_list.extend([weights_delta, bias_delta])

            total_loss_flow.cuda()

            self.manager.append('flow_loss', total_loss_flow.item())
            self.manager.append('delta_theta_std', sum(delta_params_std_list) / len(delta_params_std_list))
            if not density_loss_list:
                density_loss_list = [torch.tensor([0]).cuda()]
            self.manager.append('flow_density_loss', torch.stack(density_loss_list).mean(dtype=torch.float).item())
            if not raw_loss_flow_list:
                raw_loss_flow_list = [torch.tensor([0]).cuda()]
            self.manager.append('flow_loss_raw', torch.stack(raw_loss_flow_list).mean(dtype=torch.float).item())
            return delta_params_list, total_loss_flow

        else:
            raise NotImplementedError("Use --hm_use_class_batch_input for flow support.")

    def _update_weight(self, weight, update_value):
        if self.hm_update_operator != 'minus':
            raise NotImplementedError("flow loss formula hardcoded for minus update-operator only")

        if self.hm_update_operator == 'minus':
            if weight.fast is None:
                weight.fast = weight - update_value
            else:
                weight.fast = weight.fast - update_value
        elif self.hm_update_operator == 'plus':
            if weight.fast is None:
                weight.fast = weight + update_value
            else:
                weight.fast = weight.fast + update_value
        elif self.hm_update_operator == 'multiply':
            if weight.fast is None:
                weight.fast = weight * update_value
            else:
                weight.fast = weight.fast * update_value

    def _update_hm_maml_warmup_coef(self):
        if not self.hm_maml_warmup:
            self.hm_maml_warmup_coef = 0
            return
        if self.epoch < self.hm_maml_warmup_epochs:
            self.hm_maml_warmup_coef = 1.0
            return
        elif self.hm_maml_warmup_epochs <= self.epoch < self.hm_maml_warmup_epochs + self.hm_maml_warmup_switch_epochs:
            self.hm_maml_warmup_coef = (self.hm_maml_warmup_switch_epochs + self.hm_maml_warmup_epochs - self.epoch) / \
                                       (self.hm_maml_warmup_switch_epochs + 1)
            return
        self.hm_maml_warmup_coef = 0

    def _dkl_downfall(self):
        if self.dkl_downfall_stop_epoch == -1:
            self.dkl_downfall_stop_epoch = self.stop_epoch
        if self.dkl_downfall_strategy == 'None':
            self.flow.epoch_property.dkl_w = 1.0
            return
        if self.dkl_downfall_strategy == 'Exp':
            raise NotImplementedError()

        if self.dkl_downfall_start_epoch <= self.epoch < self.dkl_downfall_stop_epoch:
            if not self.dkl_downfall_linear_delta:
                num_epochs = self.dkl_downfall_stop_epoch - self.dkl_downfall_start_epoch
                assert num_epochs >= 0
                self.dkl_downfall_linear_delta = (1 - 10 ** (-self.dkl_downfall_magnitude) ) * \
                                                 self.flow.epoch_property.dkl_w / num_epochs
            self.flow.epoch_property.dkl_w = self.flow.epoch_property.dkl_w - self.dkl_downfall_linear_delta

    def _update_network_weights(self, delta_params_list, flow_loss, support_embeddings, support_data_labels,
                                train_stage=False):
        if self.hm_maml_warmup and not self.single_test:
            self._update_hm_maml_warmup_coef()
            if self.hm_maml_warmup_coef > 0.0:
                fast_parameters = []
                if self.hm_maml_update_feature_net:
                    fet_fast_parameters = list(self.feature.parameters())
                    for weight in self.feature.parameters():
                        weight.fast = None
                    self.feature.zero_grad()
                    fast_parameters = fast_parameters + fet_fast_parameters

                clf_fast_parameters = list(self.classifier.parameters())
                for weight in self.classifier.parameters():
                    weight.fast = None
                self.classifier.zero_grad()
                fast_parameters = fast_parameters + clf_fast_parameters

                for task_step in range(self.task_update_num):
                    scores = self.classifier(support_embeddings)

                    loss_ce = self.loss_fn(scores, support_data_labels)

                    # if self.hm_maml_warmup_coef < 1:
                    #     flow_loss = flow_loss - self.flow.get_density_loss(list(self.classifier.parameters())).to(loss_ce)

                    set_loss = loss_ce + self.flow_w * flow_loss.to(loss_ce)

                    grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True,
                                               allow_unused=True)  # build full graph support gradient of gradient

                    if self.approx:
                        grad = [g.detach() for g in
                                grad]  # do not calculate gradient of gradient if using first order approximation

                    if self.hm_maml_update_feature_net:
                        # update weights of feature networ
                        for k, weight in enumerate(self.feature.parameters()):
                            update_value = self.train_lr * self.hm_maml_warmup_coef * grad[k]
                            self._update_weight(weight, update_value)

                    classifier_offset = len(fet_fast_parameters) if self.hm_maml_update_feature_net else 0

                    if self.hm_maml_warmup_coef == 1:
                        # update weights of classifier network by adding gradient
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = (self.train_lr * grad[classifier_offset + k]).unsqueeze(0)
                            self._update_weight(weight, update_value)

                    elif 0.0 < self.hm_maml_warmup_coef < 1.0:
                        # update weights of classifier network by adding gradient and output of hypernetwork
                        for k, weight in enumerate(self.classifier.parameters()):
                            update_value = ((self.train_lr * self.hm_maml_warmup_coef * grad[classifier_offset + k]) + (
                                    (1 - self.hm_maml_warmup_coef) * delta_params_list[k]))
                            self._update_weight(weight, update_value)
            else:
                for k, weight in enumerate(self.classifier.parameters()):
                    update_value = delta_params_list[k]
                    self._update_weight(weight, update_value)
        else:
            for k, weight in enumerate(self.classifier.parameters()):
                update_value = delta_params_list[k]
                self._update_weight(weight, update_value)

        def group_layers(lst):
            """ generates entries of layers (weights + bias)"""
            for i in range(0, len(lst), 2):
                yield lst[i:i+2]

        for w, b in group_layers(list(self.classifier.parameters())):
            weights = torch.cat([w.fast, b.fast.unsqueeze(-1)], axis=2)
            self.manager.append('theta_norm', torch.linalg.vector_norm(weights, dim = (1, 2)).mean().item())


    def _get_list_of_delta_params(self, maml_warmup_used, support_embeddings, support_data_labels, train_stage):
        flow_loss = torch.tensor([0]).cuda()
        if not maml_warmup_used:
            if self.enhance_embeddings:
                with torch.no_grad():
                    logits = self.classifier.forward(support_embeddings).detach()
                    logits = F.softmax(logits, dim=1)

                labels = support_data_labels.view(support_embeddings.shape[0], -1)
                support_embeddings = torch.cat((support_embeddings, logits, labels), dim=1)

            for weight in self.parameters():
                weight.fast = None
            self.zero_grad()

            support_embeddings = self.apply_embeddings_strategy(support_embeddings)
            delta_params, flow_loss = self.get_hn_delta_params(support_embeddings, train_stage)
            if self.hm_save_delta_params and len(self.delta_list) == 0:
                self.delta_list = [{'delta_params': delta_params}]

            return delta_params, flow_loss
        else:
            return [torch.zeros(*i).cuda() for (_, i) in self.target_net_param_shapes.items()],flow_loss

    def forward(self, x):
        out = self.feature.forward(x)

        if self.hm_detach_feature_net:
            out = out.detach()

        scores = self.classifier.forward(out)
        return scores

    def set_forward(self, x, is_feature=False, train_stage=False):
        """ 1. Get delta params from hypernetwork with support data.
        2. Update target- network weights.
        3. Forward with query data.
        4. Return scores"""

        assert is_feature == False, 'MAML do not support fixed feature'

        x = x.cuda()
        x_var = Variable(x)
        support_data = x_var[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support,
                                                                            *x.size()[2:])  # support data
        query_data = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query,
                                                                          *x.size()[2:])  # query data
        support_data_labels = self.get_support_data_labels()

        support_embeddings = self.feature(support_data)

        if self.hm_detach_feature_net:
            support_embeddings = support_embeddings.detach()

        maml_warmup_used = (
                (not self.single_test) and self.hm_maml_warmup and (self.epoch < self.hm_maml_warmup_epochs))

        delta_params_list, flow_loss = self._get_list_of_delta_params(maml_warmup_used, support_embeddings,
                                                                      support_data_labels, train_stage)
        # if not flow_loss.dim() == 0:
        #     flow_loss = torch.sum(flow_loss)

        self._update_network_weights(delta_params_list, flow_loss, support_embeddings, support_data_labels, train_stage)

        if self.hm_set_forward_with_adaptation and not train_stage:
            scores = self.forward(support_data)
            return scores, None, flow_loss
        else:
            if self.hm_support_set_loss and train_stage and not maml_warmup_used:
                query_data = torch.cat((support_data, query_data))

            scores = self.forward(query_data)

            # sum of delta params for regularization
            if self.hm_lambda != 0:
                total_delta_sum = sum([delta_params.pow(2.0).sum() for delta_params in delta_params_list])

                return scores, total_delta_sum, flow_loss
            else:
                return scores, None, flow_loss

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores, total_delta_sum, flow_loss = self.set_forward(x, is_feature=False, train_stage=True)
        query_data_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()

        if self.hm_support_set_loss:
            support_data_labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).cuda()
            query_data_labels = torch.cat((support_data_labels, query_data_labels))

        loss_ce = self.loss_fn(scores, query_data_labels)

        loss = loss_ce + self.flow_w * flow_loss.to(loss_ce)

        self.manager.append('loss_ce', loss_ce.item())

        if self.hm_lambda != 0:
            loss = loss + self.hm_lambda * total_delta_sum
        self.manager.append('loss',loss)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = query_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(query_data_labels)) * 100
        self.manager.append('acc',task_accuracy)

    def set_forward_loss_with_adaptation(self, x):
        scores, _, flow_loss = self.set_forward(x, is_feature=False, train_stage=False)
        support_data_labels = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()

        loss_ce = self.loss_fn(scores, support_data_labels)
        self.manager.append('loss_ce',loss_ce.item())
        loss = loss_ce + self.flow_w * flow_loss.to(loss_ce)
        self.manager.append('loss',loss)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy().flatten()
        y_labels = support_data_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind == y_labels)
        task_accuracy = (top1_correct / len(support_data_labels)) * 100

        return loss, task_accuracy

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        self.manager.clear_all_fields()
        print_freq = 10
        task_count = 0
        optimizer.zero_grad()

        self.delta_list = []

        # train
        for i, (x, _) in enumerate(train_loader):
            if self.epoch >= self.hm_maml_warmup_epochs:
                self.flow.epoch_property.curr_epoch = self.epoch - self.hm_maml_warmup_epochs

            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            self.set_forward_loss(x)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(self.manager.loss).sum(0)
                loss_q.backward()
                optimizer.step()
                task_count = 0
                self.manager.clear_field('loss')

            optimizer.zero_grad()
            if i % print_freq == 0:
                #print(self.manager.loss)
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f}'.format(self.epoch, self.stop_epoch, i,
                                                                             len(train_loader),
                                                                             torch.stack(self.manager.loss).sum(0).item() / float(i + 1)))
        if self.hm_maml_warmup_coef < 1:
            self._sample_temp_step()
            self._dkl_downfall()
            #print(f"Epoch {self.epoch}; F_epoch: {self.flow.epoch_property.curr_epoch}: dkl_w {self.flow.epoch_property.dkl_w}, "
            #f"temp_w {self.flow.epoch_property.temp_w}")

        metrics = self.manager.get_metrics(clean_after=True)

        if self.hn_adaptation_strategy == 'increasing_alpha':
            metrics['alpha'] = self.alpha

        if self.hm_save_delta_params and len(self.delta_list) > 0:
            delta_params = {"epoch": self.epoch, "delta_list": self.delta_list}
            metrics['delta_params'] = delta_params

        if self.alpha < 1:
            self.alpha += self.hn_alpha_step

        return metrics

    def test_loop(self, test_loader, return_std=False, return_time: bool = False):  # overwrite parrent function
        self.manager.clear_all_fields()
        acc_all = []
        self.delta_list = []
        acc_at = defaultdict(list)

        iter_num = len(test_loader)

        eval_time = 0

        if self.hm_set_forward_with_adaptation:
            for i, (x, _) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(0), "MAML do not support way change"
                s = time()
                acc_task, acc_at_metrics = self.set_forward_with_adaptation(x)
                t = time()
                for (k, v) in acc_at_metrics.items():
                    acc_at[k].append(v)
                acc_all.append(acc_task)
                eval_time += (t - s)

        else:
            for i, (x, _) in enumerate(test_loader):
                self.n_query = x.size(1) - self.n_support
                assert self.n_way == x.size(0), f"MAML do not support way change, {self.n_way=}, {x.size(0)=}"
                s = time()
                correct_this, count_this, loss_ce_test = self.correct(x)
                t = time()
                acc_all.append(correct_this / count_this * 100)
                eval_time += (t - s)

        metrics = {
            k: np.mean(v) if len(v) > 0 else 0
            for (k, v) in acc_at.items()
        }
        if not self.hm_set_forward_with_adaptation:
            metrics["ce_loss_test"] = loss_ce_test.item()

        num_tasks = len(acc_all)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        print("Num tasks", num_tasks)

        ret = [acc_mean]
        if return_std:
            ret.append(acc_std)
        if return_time:
            ret.append(eval_time)
        ret.append(metrics)

        return ret

    def set_forward_with_adaptation(self, x: torch.Tensor):
        self_copy = deepcopy(self)

        # deepcopy does not copy "fast" parameters so it should be done manually
        for param1, param2 in zip(self.parameters(), self_copy.parameters()):
            if hasattr(param1, 'fast'):
                if param1.fast is not None:
                    param2.fast = param1.fast.clone()
                else:
                    param2.fast = None

        metrics = {
            "accuracy/val@-0": self_copy.query_accuracy(x)
        }

        val_opt_type = torch.optim.Adam if self.hn_val_optim == "adam" else torch.optim.SGD
        val_opt = val_opt_type(self_copy.parameters(), lr=self.hn_val_lr)

        if self.hn_val_epochs > 0:
            for i in range(1, self.hn_val_epochs + 1):
                self_copy.train()
                val_opt.zero_grad()
                loss, val_support_acc = self_copy.set_forward_loss_with_adaptation(x)
                loss.backward()
                val_opt.step()
                self_copy.eval()
                metrics[f"accuracy/val_support_acc@-{i}"] = val_support_acc
                metrics[f"accuracy/val_loss@-{i}"] = loss.item()
                metrics[f"accuracy/val@-{i}"] = self_copy.query_accuracy(x)

        # free CUDA memory by deleting "fast" parameters
        for param in self_copy.parameters():
            param.fast = None

        return metrics[f"accuracy/val@-{self.hn_val_epochs}"], metrics

    def query_accuracy(self, x: torch.Tensor) -> float:
        scores, *_ = self.set_forward(x, train_stage=True)
        return 100 * accuracy_from_scores(scores, n_way=self.n_way, n_query=self.n_query)

    def get_logits(self, x):
        self.n_query = x.size(1) - self.n_support
        logits, *_ = self.set_forward(x)
        return logits

    def correct(self, x):
        scores, *_ = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        with torch.no_grad():
            loss_ce = self.loss_fn(scores, torch.from_numpy(y_query).cuda())

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query), loss_ce
