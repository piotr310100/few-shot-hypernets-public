import torch
import numpy as np
from torch import optim
from torch import nn
from .flow import get_point_cnf
from .flow import get_hyper_cnf
from ..utils import truncated_normal, standard_normal_logprob, standard_laplace_logprob
from torch.nn import init
from torch.distributions.laplace import Laplace


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None):
    if padding is None:
        padding_inside = (kernel_size - 1) // 2
    else:
        padding_inside = padding
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_inside,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_inside, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=bias),
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i, j, :, :] = torch.from_numpy(bilinear)


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class CRegression(nn.Module):
    class EpochManager:
        def __init__(self, num_zeros_warmup_epochs):
            # changes via fhypermaml module
            self.temp_w = 0
            self.dkl_w = 0
            self.curr_epoch = 0
            self.num_zeros_warmup_epochs = num_zeros_warmup_epochs

        def is_zero_warmup_epoch(self):
            return self.curr_epoch < self.num_zeros_warmup_epochs

    def __init__(self, args):
        super(CRegression, self).__init__()
        self.input_dim = args.input_dim
        self.hn_shape = torch.Size([args.shape1, args.shape2])
        self.flownet = FlowNetS()
        self.args = args
        self.point_cnf = get_point_cnf(args)
        self.gpu = args.gpu
        self.logprob_type = args.logprob_type
        self.const_sample = self.sample_gaussian((self.hn_shape[0] * self.hn_shape[1], 1), None, self.gpu)
        self.prior_distribution = torch.distributions.MultivariateNormal(
            torch.zeros(self.hn_shape[0] * self.hn_shape[1]).cuda(),
            torch.eye(self.hn_shape[0] * self.hn_shape[1]).cuda())
        self.epoch_property = self.EpochManager(args.num_zeros_warmup_epochs)
        self.dim_reducer_hn = torch.nn.Linear(self.hn_shape[0] * self.hn_shape[1], args.zdim)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer

        opt = _get_opt_(list(self.flownet.parameters()) + list(self.point_cnf.parameters()))
        return opt

    def get_sample(self, num_points):
        y = self.epoch_property.temp_w * self.sample_gaussian((1, num_points, self.hn_shape[0] * self.hn_shape[1]), None, self.gpu)
        return y

    # def get_fast_weights(self, global_weight, delta_weight):
    #     return torch.cat([global_weight[0], global_weight[1].reshape(-1, 1)], axis=1) - delta_weight.reshape \
    #         (*self.hn_shape)
    #
    def get_density_loss(self, delta_weights:torch.tensor):
        """gets density loss for the output of hypernetwork"""
        return self.prior_distribution.log_prob(delta_weights).mean()

    # def get_density_loss(self, weights):
    #     if weights[0].fast is not None:
    #         return self.prior_distribution.log_prob(torch.cat([weights[0].fast, weights[1].fast.reshape(-1, 1)], axis=1).flatten())
    #     return torch.tensor([0])

    def forward(self, x: torch.tensor, train_stage = False, norm_warmup = False):  # x.shape = 5,65 = 5,64 + bias
        # 1) output z hypernetworka zamieniamy na embedding rozmiaru z_dim
        z = x.flatten()
        # 2) wylosuj sample z rozkladu normalnego
        num_points = self.num_points
        y = self.get_sample(num_points)
        # 3) przerzuc przez flow -> w_i := F_{\theta}(z_i)
        z = self.dim_reducer_hn(z).reshape(-1)
        delta_target_networks_weights = self.point_cnf(y, z, reverse=True).view(*y.size())
        delta_target_networks_weights = delta_target_networks_weights.reshape(num_points, -1)

        return delta_target_networks_weights

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def sample_laplace(size, gpu=None):
        m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        y = m.sample(sample_shape=torch.Size([size[0], size[1], size[2]])).float().squeeze(3)
        y = y if gpu is None else y.cuda(gpu)
        return y

    def decode(self, z, num_points):
        # transform points from the prior to a point cloud, conditioned on a shape code
        w = self.flownet(z)
        if self.logprob_type == "Laplace":
            y = self.sample_laplace((z.size(0), num_points, self.input_dim), self.gpu)
        if self.logprob_type == "Normal":
            y = self.sample_gaussian((z.size(0), num_points, self.input_dim), None, self.gpu)
        x = self.point_cnf(y, w, reverse=True).view(*y.size())
        return y, x

    def get_logprob(self, x, y_in):
        batch_size = x.size(0)
        w = self.flownet(x)

        # Loss
        y, delta_log_py = self.point_cnf(y_in, w, torch.zeros(batch_size, y_in.size(1), 1).to(y_in))
        if self.logprob_type == "Laplace":
            log_py = standard_laplace_logprob(y)
        if self.logprob_type == "Normal":
            log_py = standard_normal_logprob(y)

        batch_log_py = log_py.sum(dim=2)
        batch_log_px = batch_log_py - delta_log_py.sum(dim=2)
        log_py = log_py.view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, y.size(1), 1).sum(1)
        log_px = log_py - delta_log_py

        return log_py, log_px, (batch_log_py, batch_log_px)


class FlowNetS(nn.Module):
    def __init__(self, input_channels=12, batchNorm=True):
        super(FlowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.conv7 = conv(self.batchNorm, 1024, 1024, kernel_size=1, stride=1, padding=0)
        self.conv8 = conv(self.batchNorm, 1024, 1024, kernel_size=1, stride=1, padding=0)
        # self.fc1 = nn.Linear(in_features=46080, out_features=1024, bias=True)
        self.fc1 = nn.Linear(in_features=46080, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024, bias=True)
        # self.predict_6 = predict_flow(1024)
        # self.fc1 = nn.Linear(in_features=90, out_features=512, bias=True)
        # self.fc2 = nn.Linear(in_features=512, out_features=1024, bias=True)
        # self.fc3 = nn.Linear(in_features=1024, out_features=2048, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        out_conv8 = self.conv8(self.conv7(out_conv6))
        # predict = self.predict_6(out_conv8)
        # out_fc1 = nn.functional.relu(self.fc1(predict.view(predict.size(0), -1)))
        # out_fc2 = nn.functional.relu(self.fc2(out_fc1))
        # out_fc3 = nn.functional.relu(self.fc3(out_fc2))
        # out_conv7 = self.conv7(out_conv6)
        # out_conv8 = self.conv8(out_conv7)
        # out_fc2 = out_conv6.view(out_conv6.size(0), -1)
        # out_fc2 = self.fc1(out_conv8.view(out_conv8.size(0), -1))
        out_fc1 = nn.functional.relu(self.fc1(out_conv8.view(out_conv6.size(0), -1)))
        out_fc2 = nn.functional.relu(self.fc2(out_fc1))
        # out_fc2 = self.predict_6(out_conv6)
        return out_fc2
