import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Hardtanh, BatchNorm1d as BN
from torch.nn.modules.utils import _single
from torch.autograd import Function
from torch.nn import Parameter
import math
import numpy as np
from torch.nn.modules.linear import Linear
import time
from torch.autograd import Function, Variable
from _binary_base_plus import _Conv2dB, Qmodes, _LinearB, _ActB, _ActB_qk

__all__ = ['BinaryActivation', 'LearnableBias', 'MeanShift', 'BiLinearBiReal', 'LinearBi', 'ActBi']


activations = {
    'ReLU': ReLU,
    'Hardtanh': Hardtanh
}

        
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(-0.5 * torch.ones(1,1,out_chn,out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class MeanShift(torch.nn.Module):

    def __init__(self, channels):
        super(MeanShift, self).__init__()
        # self.register_buffer('median', torch.zeros((1, channels)))
        self.register_buffer('median', torch.zeros((0.5, channels)))
        self.register_buffer("num_track", torch.LongTensor([0]))

    def forward(self, x):
        if self.training:
            median = torch.sort(x, dim=0)[0][x.shape[0] // 2].view(1, -1)
            self.median.mul_(self.num_track)
            self.median.add_(median)
            self.median.div_(self.num_track + 1)
            self.num_track.add_(1)
            self.median.detach_()
            self.num_track.detach_()
            x = x - self.median
        else:
            x = x - self.median
        return x
        

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class BiLinearBiReal(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(BiLinearBiReal, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

    def forward(self, input):
        x = input
        out_forward = torch.sign(input)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        input = out_forward.detach() - out3.detach() + out3
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        output = F.linear(input, binary_weights)
        return output

class BiConv2dBiReal(torch.nn.Conv2d):
    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=0, groups=1, binary_act=True):
        super(BiConv2dBiReal, self).__init__(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.binary_act = binary_act

    def forward(self, input):
        x = input
        out_forward = torch.sign(input)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        input = out_forward.detach() - out3.detach() + out3
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        output = F.conv2d(input, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)
        return output

def sign_pass(x):
    y = torch.sign(x)
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class LinearBi(_LinearB):
    def __init__(self, in_features, out_features, bias=True, nbits_w=1, **kwargs):
        super(LinearBi, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        self.act = ActBi(in_features=in_features, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -1
        Qp = 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1)
        w_q = sign_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        x = self.act(x)

        return F.linear(x, w_q, self.bias)

class ActBi(_ActB):
    def __init__(self, in_features, nbits_a=1, mode=Qmodes.kernel_wise, **kwargs):
        super(ActBi, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        # print(self.alpha.shape, self.zero_point.shape)
    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            Qn = -1
            Qp = 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(torch.mean(abs(x),dim=1,keepdim=True))
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
            self.init_state.fill_(1)
        

        Qn = -1
        Qp = 1

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scale(self.alpha, g)
        zero_point = grad_scale(zero_point, g)

        if len(x.shape)==2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = sign_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha

        return x