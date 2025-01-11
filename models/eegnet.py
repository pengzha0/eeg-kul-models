import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim




class EEGNet(nn.Module):
    def __init__(self, channels_num):
        super(EEGNet, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv1d(channels_num, 64, 3, padding=3)
        self.batchnorm1 = nn.BatchNorm1d(64, False)

        # Layer 2
        # self.channel_att1 = Channel_Attention(256)


        self.conv2 =nn.Conv1d(64, 128, 7, padding=3)
        self.batchnorm2 = nn.BatchNorm1d(128, False)
        self.pooling2 = nn.MaxPool1d(4, 4)

        # Layer 3
        # self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv1d(128, 128, 7, padding=3)
        self.batchnorm3 = nn.BatchNorm1d(128, False)
        self.pooling3 = nn.MaxPool1d(4, 4)

        self.act  = nn.PReLU()
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        # self.attn = Channel_Attention(32)

        self.lr = nn.Linear(128, 2)

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
        return lr

if __name__ == '__main__':
    model = EEGNet().cuda()
    data = torch.zeros((16, 50, 4560)).cuda()
    out = model(data)
    print(out.shape)












