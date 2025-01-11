import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# setting
time_len = 1

# model setting
is_space_attention = True
is_temporal_attention = True
lr = 5e-3
epochs = 300
sample_len, channels_num = int(128 * time_len), 64
cnn_kernel_num = 64
cnn_block_len = 4

# space attention setting
sa_kq = 50
sa_block_num = math.ceil(time_len)
sa_channel_dense_num = cnn_kernel_num * sa_block_num

# temporal attention setting
ta_do_percent = 0.5

class MySpaceAttention(nn.Module):
    def __init__(self):
        super(MySpaceAttention, self).__init__()
        se_cnn_num = 10
        self.conv = nn.Conv2d(1, se_cnn_num, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.max_pool = nn.MaxPool2d((1, se_cnn_num))
        self.dropout1 = nn.Dropout(ta_do_percent)
        self.fc1 = nn.Linear(channels_num, 8)
        self.elu = nn.ELU()
        self.dropout2 = nn.Dropout(ta_do_percent)
        self.fc2 = nn.Linear(8, channels_num)
    
    def forward(self, x):
        x = x.view(-1, int(sample_len / sa_block_num), sa_block_num, channels_num)
        temp_x = x.permute(0, 2, 1, 3)

        x = self.conv(temp_x)
        x = x.permute(0, 3, 2, 1)
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 3, 1)  # Permute back to original shape for the linear layers
        # x = x.view(x.size(0), x.size(1), -1)  # Flatten the last two dimensions for the linear layer
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout2(x)
        w = self.fc2(x)
        w = self.elu(w)
        temp_x = temp_x.permute(0, 2, 1, 3)
        
        # SE attention
        y = w * temp_x
        y = y.view(-1, channels_num,sample_len )

        return y

class MyTemporalAttention(nn.Module):
    def __init__(self):
        super(MyTemporalAttention, self).__init__()
        self.dense_k = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(sa_block_num * cnn_kernel_num, sa_kq),
            nn.ELU(),
        )
        self.dense_q = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(sa_block_num * cnn_kernel_num, sa_kq),
            nn.ELU(),
        )
        self.dense_v = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(sa_block_num * cnn_kernel_num, sa_channel_dense_num),
            nn.Tanh(),
        )
        self.my_se_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0,2,1)
        k = self.dense_k(x)
        q = self.dense_q(x)
        v = self.dense_v(x)
        w = torch.bmm(k, q.transpose(1, 2)) / math.sqrt(sample_len)
        w = self.my_se_softmax(w)
        y = torch.bmm(w, v)

        return y

class StaNet(nn.Module):
    def __init__(self, sample_len=128, channels_num=64, is_attention=True):
        super(StaNet, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(channels_num)
        self.space_attention = MySpaceAttention() if is_space_attention else nn.Identity()
        self.conv1 = nn.Conv1d(channels_num, cnn_kernel_num, kernel_size=5, stride=1, padding='same')
        self.max_pool = nn.MaxPool1d(cnn_block_len)
        self.temporal_attention = MyTemporalAttention() if is_attention else nn.Identity()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cnn_kernel_num * sa_block_num * int(sample_len / cnn_block_len / sa_block_num), 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to match BatchNorm1d expected input
        x = self.batch_norm1(x)
        x = self.space_attention(x)
        x = F.tanh(self.conv1(x))
        x = self.max_pool(x)
        x = x.view(x.size(0), sa_block_num * cnn_kernel_num,int(sample_len / cnn_block_len / sa_block_num))
        x = self.temporal_attention(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.softmax(self.fc(x), dim=1)
        return x

def create_model(sample_len=128, channels_num=64, lr=1e-3, is_attention=True):
    model = CustomModel(sample_len, channels_num, is_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def main():
    sample_len = 128
    channels_num = 64
    model, optimizer, criterion = create_model(sample_len, channels_num)
    random_data = torch.ones((16, sample_len, channels_num))
    output = model(random_data)
    print(model)
    del model

if __name__ == '__main__':
    main()