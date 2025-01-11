import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class BNNConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(BNNConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size
        self.shape = (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.rand(*self.shape) * 0.001, requires_grad=True)

    def forward(self, x):

        binary_input_no_grad = torch.sign(x)
        cliped_input = torch.clamp(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = self.weight.view(self.shape)
        binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.2, 1.2)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

class Channel_Attention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.LeakyReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
class EEGNet_bnn(nn.Module):
    def __init__(self, channels_num):
        super(EEGNet_bnn, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv1d(channels_num, 64, 3, padding=3)
        self.batchnorm1 = nn.BatchNorm1d(64, False)

        # Layer 2
        # self.channel_att1 = Channel_Attention(256)


        self.conv2 =BNNConv1d(64, 256, 7, padding=3)
        self.batchnorm2 = nn.BatchNorm1d(256, False)
        self.pooling2 = nn.MaxPool1d(4, 4)

        # Layer 3
        # self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = BNNConv1d(256, 256, 7, padding=3)
        self.batchnorm3 = nn.BatchNorm1d(256, False)
        self.pooling3 = nn.MaxPool1d(4, 4)

        self.act  = nn.PReLU()
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        # self.attn = Channel_Attention(32)

        self.lr = nn.Linear(256, 2)
        self.angle = nn.Linear(256, 5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(x.shape)# torch.Size([16, 4560, 50])
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act(x)
        # x = F.dropout(x, 0.3)
        # print(x.shape)# torch.Size([16, 256, 50])
        # x = x.permute(0, 3, 1, 2)

        # Layer 2
        # x = self.padding1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.act(x)
        # x = F.dropout(x, 0.3)
        x = self.pooling2(x)
        # print(x.shape)# torch.Size([16, 256, 12])

        # Layer 3
        # x = self.padding2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act(x)
        # x = F.dropout(x, 0.3)
        x = self.pooling3(x)
        # print(x.shape)# torch.Size([16, 256, 3])

        # x = self.channel_att1(x)
        # FC Layer
        x = x.mean(-1)
        lr = self.lr(x)
        angle = self.angle(x)
        return lr,angle

if __name__ == '__main__':
    model = EEGNet().cuda()
    data = torch.zeros((16, 50, 4560)).cuda()
    out = model(data)
    print(out.shape)












