# import numpy as np
# import torch
import torch.nn as nn
from torch.nn import functional as F  # ？
# from utis import *
from Liblinks.utis import *
from torch.nn.modules.utils import _pair
from torch.nn import init


# from utis import max_singular_value
# from utis import _l2normalize


#
class ConditionBatchNorm2d(nn.BatchNorm2d):  # 条件BN层
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ConditionBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        output = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2dAttr(ConditionBatchNorm2d):
    def __init__(self, num_attri, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(CategoricalConditionalBatchNorm2dAttr, self).__init__(num_features, eps, momentum, affine,
                                                                    track_running_stats)
        self.weights = SNLinear(num_attri, num_features)
        self.biases = SNLinear(num_attri, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, a, **kwargs):
        weight = self.weights(a)
        bias = self.biases(a)
        return super(CategoricalConditionalBatchNorm2dAttr, self).forward(input, weight, bias)


class CategoricalConditionalBatchNorm2dEmbed(ConditionBatchNorm2d):
    def __init__(self, num_cls, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(CategoricalConditionalBatchNorm2dEmbed, self).__init__(num_features, eps, momentum, affine,
                                                                     track_running_stats)
        self.weights = SNEmbedCLS(num_cls, num_features)
        self.biases = SNEmbedCLS(num_cls, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        # init.zeros_(self.biases.weight.data)
        init.ones_(self.biases.weight.data)

    def forward(self, input, a, **kwargs):
        weight = self.weights(a)
        bias = self.biases(a)
        return super(CategoricalConditionalBatchNorm2dEmbed, self).forward(input, weight, bias)


class SNConv2d(nn.Conv2d):  # 2维卷积层
    def __init__(self, in_channels: int,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 use_gamma=False,
                 Ip=1,
                 factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                       padding_mode)
        self.u = np.random.normal(size=(1, out_channels)).astype(dtype="f")

    def W_bar(self):
        W_mat = self.weight.data.reshape(self.weight.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        if self.training:
            self.u[:] = _u
        # self.u[:]=_u
        return self.weight / sigma

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.W_bar(), self.bias)


class SNLinear(nn.Linear):  # 线性全连接层
    def __init__(self, in_features, out_features, bias=True, use_gamma=False, Ip=1, factor=None):
        self.Ip = Ip
        self.use_gamma = use_gamma
        self.factor = factor
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = np.random.normal(size=(1, out_features)).astype(dtype="f")

    def W_bar(self):
        W_mat = self.weight.data.reshape(self.weight.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        # sigma=torch.from_numpy(sigma)
        # print(sigma)
        # print(_u)
        if self.training:
            self.u[:] = _u
        # self.u[:]=_u
        # print(self.weight)
        # print(self.weight/sigma)
        return self.weight / sigma  # 如果出问题检查这里的梯度反传

    def forward(self, input):
        return F.linear(input, self.W_bar(), self.bias)


class SNEmbedCLS(nn.Embedding):  # embedding层做查询表
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, Ip=1, factor=None):
        self.Ip = Ip
        self.factor = factor
        super(SNEmbedCLS, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                                         scale_grad_by_freq, sparse, _weight)

        self.u = np.random.normal(size=(1, num_embeddings)).astype(dtype="f")

    def W_bar(self):
        W_mat = self.weight.data.reshape(self.weight.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        if self.training:
            self.u[:] = _u
        return self.weight / sigma

    def forward(self, input):
        return F.embedding(input, self.W_bar(), self.padding_idx, self.max_norm, self.norm_type,
                           self.scale_grad_by_freq, self.sparse)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.Hinge = nn.ReLU(inplace=True)

    def forward(self, loss_dcri, loss_gen):
        loss = self.Hinge(1 - loss_dcri).mean()
        loss += self.Hinge(1 + loss_gen).mean()
        return loss
