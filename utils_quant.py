import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class ActLSQ_bi(nn.Module):
    def __init__(self, in_features, **kwargs):
        super(ActLSQ_bi, self).__init__()
        # print(self.alpha.shape, self.zero_point.shape)
        self.alpha = Parameter(torch.Tensor(in_features))
        self.zero_point = Parameter(torch.Tensor(in_features))
        torch.nn.init.zeros_(self.zero_point)

    def forward(self, x):
        if self.alpha is None:
            return x

        '''if self.training and self.init_state == 0:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            # print(self.signed)
            Qn = -1
            Qp = 1
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.zero_point.data.copy_(self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
            self.init_state.fill_(1)'''
        
        # print(self.signed)

        Qn = -1
        Qp = 1
        g = 1.0 / math.sqrt(x.numel() * Qp)

        zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        alpha = grad_scale(self.alpha, g)
        zero_point = grad_scale(zero_point, g)

        if len(x.shape)==2:
            alpha = alpha.unsqueeze(0)
            zero_point = zero_point.unsqueeze(0)
        elif len(x.shape)==4:
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        x = (x - zero_point) * alpha
        return x


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input

class BinaryQuantizerMCN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, origin_weight, weight, MFilter):
        
        #  MFilter = torch.abs(MFilter)

        #bin = 0.02
        # MFilterMean_temp=torch.sum(MFilter, dim=1)
        # MFilterMean = torch.sum(MFilterMean_temp, dim=1)
        # scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
        # scaling_factor = scaling_factor.detach()
        # real_weights = weight - torch.mean(weight, dim=-1, keepdim=True)
        # binary_weights_no_grad = MFilter * torch.sign(real_weights)
        # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        # out = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        # weight_bin = torch.sign(weight)# * bin
        # out = torch.sign(weight) * MFilter
        ctx.save_for_backward(origin_weight, weight, MFilter)
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        origin_weight, weight, MFilter = ctx.saved_tensors

        para_loss = 0.0001
        #bin = 0.02
        # real_weights = origin_weight - torch.mean(origin_weight, dim=-1, keepdim=True)
        # binary_weights_no_grad = MFilter * torch.sign(real_weights)
        # cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        # out = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        
        
        weight_bin = torch.sign(weight)

        # target1 = para_loss * (origin_weight - weight_bin * MFilter)       
        gradWeight = grad_output # * MFilter


        target2 = (origin_weight - weight_bin * MFilter) * weight_bin
        grad_h2_sum = torch.sum(grad_output * origin_weight, keepdim=True, dim=1)
        
        grad_target2 = torch.sum(target2, keepdim=True, dim=1)
        gradMFilter = grad_h2_sum  - para_loss * grad_target2
        # gradOrigin_weight = 0

        return None, gradWeight, gradMFilter

class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out==-1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(self,  *kargs,bias=True, config=None, type=None):
        super(QuantizeLinear, self).__init__(*kargs,bias=True)
        self.quantize_act = config.quantize_act
        self.weight_bits = config.weight_bits
        self.quantize_act = config.quantize_act
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        self.init = True
            
        if self.quantize_act:
            self.input_bits = config.input_bits
            if self.input_bits == 1:
                self.act_quantizer = BinaryQuantizer
            elif self.input_bits == 2:
                self.act_quantizer = TwnQuantizer
            else:
                self.act_quantizer = SymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
        self.register_parameter('scale', Parameter(torch.Tensor([0.0]).squeeze()))
 
    def reset_scale(self, input):
        bw = self.weight
        ba = input
        self.scale = Parameter((ba.norm() / torch.sign(ba).norm()).float().to(ba.device))

    def forward(self, input, type=None):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)

        if self.input_bits == 1:
            binary_input_no_grad = torch.sign(input)
            cliped_input = torch.clamp(input, -1.0, 1.0)
            ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        else:
            ba = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        
        out = nn.functional.linear(ba, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out

class QuantizeLinearMCN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantizeLinearMCN, self).__init__(in_features, out_features, bias=True)

        self.quantize_act = True
        # self.weight_bits = config.weight_bits
        # self.quantize_act = config.quantize_act
        self.init_state = 0
        self.clip_val = 2.5

        self.MFilters = Parameter(torch.randn(self.out_features, 1))
        self.weight_quantizer = BinaryQuantizerMCN
        self.register_buffer('weight_clip_val', torch.tensor([-self.clip_val, self.clip_val]))
        self.init = True

        if self.quantize_act:
            self.act_quantizer = BinaryQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-self.clip_val, self.clip_val]))
        self.register_parameter('scale', Parameter(torch.Tensor([0.0]).squeeze()))
 
    def reset_scale(self, input):
        bw = self.weight
        ba = input
        self.MFilters = Parameter((ba.norm() / torch.sign(ba).norm()).float().to(ba.device))

    def forward(self, input):
        # scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
        # scaling_factor = scaling_factor.detach()
        if self.training and self.init_state == 0:
            self.MFilters.data.copy_(torch.mean(abs(self.weight), dim=1, keepdim=True))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state = 1   # fill_(1)

        real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
        binary_weights_no_grad = torch.abs(self.MFilters) * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # print(self.MFilters)
        weight = self.weight_quantizer.apply(self.weight, weight, torch.abs(self.MFilters))
        
        binary_input_no_grad = torch.sign(input)
        cliped_input = torch.clamp(input, -1.0, 1.0)
        ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        
        
        out = nn.functional.linear(ba, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(self,  *kargs,padding_idx=None, config=None, type=None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.init = True
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input, type=None):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, self.layerwise)
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out
