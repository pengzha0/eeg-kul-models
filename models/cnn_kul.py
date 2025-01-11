import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# setting
time_len = 1
sample_len, channels_num = int(128 * time_len), 32
lr = 5e-3

class CNN_KUL(nn.Module):
    def __init__(self, sample_len=128, channels_num=64):
        super(CNN_KUL, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(17, channels_num))
        # self.bn1 = nn.BatchNorm1d(5)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 2)
        
    
    def forward(self, x):
        x = x.view(-1, 1, x.shape[1], x.shape[2])  # Reshape to (batch_size, 1, sample_len, channels_num)
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), x.size(1), -1)  # Reshape to (batch_size, channels_num, new_sample_len)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
