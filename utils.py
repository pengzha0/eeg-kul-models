from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
from functools import partial


######################################
#       measurement functions        #
######################################

count_flops = 0
count_bops = 0
count_params = 0


def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)
def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost
def mmd_loss(source_features, target_features):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]

    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
    )

    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable
        
def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_flops, count_bops, count_params
    delta_flops = 0
    delta_bops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    #print(type_name)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_bops = 0
        delta_params = get_layer_param(layer)

    ### ops_bconv
    elif type_name in ['BConv']:
        out_h = int((x.size()[2] + 2 * layer.padding - layer.kernel_size) / \
                layer.stride + 1)
        out_w = int((x.size()[3] + 2 * layer.padding - layer.kernel_size) / \
                layer.stride + 1)
        delta_flops = 0
        delta_bops = layer.in_channels * layer.out_channels * layer.kernel_size * \
                layer.kernel_size * out_h * out_w / layer.groups * multi_add
        if layer.doublebops:
            delta_bops *= 2
        delta_params = get_layer_param(layer) / 32.0

    ### ops_transfer
    elif type_name in ['Sign', 'Identity', 'Shuffle']:
        delta_flops = 0
        delta_bops = 0
        delta_params = 0

    ### ops_nonlinearity
    elif type_name in ['BatchNorm2d', 'ReLU', 'PReLU', 'FPReLU', 'RPReLU', 'FRPReLU', 'ReLU6', 'Sigmoid']:
        """In BNN, such operation takes flops"""
        delta_flops = x.numel()
        #delta_flops = 0
        delta_bops = 0
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d', 'MaxPool2d']:
        """ AvgPool can be done by sum(.)/(kernel_ops)"""
        _, in_channels, in_w, in_h = x.size()
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_h + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_flops = in_channels * out_w * out_h
        #delta_flops = in_channels * out_w * out_h * kernel_ops
        delta_bops = 0
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        """ AvgPool can be done by sum(.)/(kernel_ops)"""
        in_w = x.size()[2]
        kernel_size = in_w
        padding = 0
        kernel_ops = kernel_size * kernel_size
        out_w = int((in_w + 2 * padding - kernel_size) / 1 + 1)
        out_h = int((in_w + 2 * padding - kernel_size) / 1 + 1)
        delta_flops = x.size()[1] * out_w * out_h
        #delta_flops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_bops = 0
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        try:
            bias_ops = layer.bias.numel()
        except AttributeError:
            bias_ops = 0
        delta_flops = x.size()[0] * (weight_ops + bias_ops)
        delta_bops = 0
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['Dropout2d', 'DropChannel', 'Dropout']:
        delta_flops = 0
        delta_bops = 0
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_flops += delta_flops
    count_bops += delta_bops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_flops, count_params, count_bops
    count_params = 0
    count_flops = 0
    count_bops = 0
    data = Variable(torch.rand(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_params, count_flops, count_bops



######################################
#         basic functions            #
######################################


class CrossEntropyLabelSmooth(nn.Module):
    """
        label smooth
    """
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def load_checkpoint(args, running_file):

    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate == 'best':
        model_filename = os.path.join(model_dir, 'model_best.pth.tar')
    elif args.evaluate is not None:
        model_filename = args.evaluate
    elif os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    running_file.write('%s\n%s\n' % (loadinfo, loadinfo2))
    running_file.flush()

    return state


def save_checkpoint(state, epoch, root, is_best, saveID, keep_freq=10):

    filename = 'checkpoint_%03d.pth.tar' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write new checkpoint 
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    # update best model 
    if is_best:
        best_filename = os.path.join(model_dir, 'model_best.pth.tar')
        shutil.copyfile(model_filename, best_filename)

    # remove old model
    if saveID is not None and (saveID + 1) % keep_freq != 0:
        filename = 'checkpoint_%03d.pth.tar' % saveID
        model_filename = os.path.join(model_dir, filename)
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print('=> removed checkpoint %s' % model_filename)

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, accum='mean'):
        self.reset()
        self.accum = accum

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.accum == 'mean':
            self.sum += val * n
            self.val = val
        elif self.accum == 'sum':
            self.sum += val
            self.val = val / n
        self.count += n
        self.avg = self.sum / self.count
        self.avg100 = self.sum / self.count * 100
        self.val100 = self.val * 100



def clip(optimizer, bound=1.2):
    """
        dim=0   clip real weights (nn.Conv2d and nn.Linear)
        dim=2   clip weights w.r.t binary weights
    """

    bound = bound * 0.999
    for i, group in enumerate(optimizer.param_groups):
        if i == 0:
            for p in group['params']:
                p.data.clamp_(-1.2, 1.2)
        elif i == 2:
            for p in group['params']:
                p.data.clamp_(-bound, bound)

def adjust_learning_rate(optimizer, epoch, args, method='cosine'):
    if method == 'cosine':
        T_total = float(args.epochs)
        T_cur = float(epoch)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))

    elif method == 'multistep':
        lr = args.lr
        for epoch_step, lr_gamma in zip(args.lr_steps, args.lr_gammas):
            if epoch >= epoch_step:
                lr = lr * lr_gamma
    if epoch < args.warm_epoch:
        lr = args.lr * (epoch + 1) / args.warm_epoch
    str_lr = ''
    for param_group, lr in zip(optimizer.param_groups, [lr, lr]):
        param_group['lr'] = lr
        str_lr = '%s-%.6f' % (str_lr, lr)
    # remove the first '-'
    return str_lr[1:]


def accuracy(output, target, topk=(1,)):
    """
        Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        #res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k)
    return res

def unionlabel(lr,angle):
    return angle*2+lr

def accuracy_asa(output1,output2, target, topk=(1,)):
    """
        Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred1 = output1.topk(maxk, 1, True, True)
    _, pred2 = output2.topk(maxk, 1, True, True)

    pred = unionlabel(pred1,pred2).t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        #res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k)
    return res


######################################
#         debug functions            #
######################################

def change_checkpoint(state):
    """
        an interface to modify the checkpoint
    """
    state_new = dict()
    for k, v in state.items():
        if 'binary_conv' in k:
            state_new[k.replace('binary_conv', 'bconv')] = v
        elif 'bn1' in k:
            state_new[k.replace('bn1', 'bn')] = v
        else:
            state_new[k] = v
    return state_new

def visualize(checkpoint, img_dir):

    from matplotlib import pyplot as plt
    import numpy as np

    state = checkpoint['state_dict']
    epoch = checkpoint['epoch']
    os.makedirs(img_dir, exist_ok=True)
    img_file = os.path.join(img_dir, 'img_epoch_%03d.png' % epoch)
    print('processing %s' % img_file)

    data = []
    for k, v in state.items():
        if 'bconv' in k and 'weights' in k:
            data.append(v.data.view(-1))

    data = torch.cat(data).cpu().numpy()

    bins = list(np.linspace(-1.5, 1.5, 200))
    plt.hist(data, bins)
    plt.savefig(img_file)
    plt.close()

    print('done')


