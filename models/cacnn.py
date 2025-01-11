import torch
import torch.nn as nn
import torch.nn.functional as F

# setting
time_len = 1
sample_len, channels_num = int(128 * time_len), 32
lr = 5e-3

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

class CACNN(nn.Module):
    def __init__(self, channels_num, sample_len, is_attention):
        super(CACNN, self).__init__()
        self.conv1 = nn.Conv1d(channels_num, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.is_attention = is_attention
        if is_attention:
            self.attn = Channel_Attention(32)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # print(x[0,0,0])
        x = F.relu(self.bn3(self.conv3(x)))
        if self.is_attention:
            x = self.attn(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.softmax(self.fc(x), dim=1)
        return x

if __name__ == '__main__':
    model = Model(channels_num=64, sample_len=128, is_attention=False)
    model = model.cuda()
    random_data = torch.ones((16, 64, 128)).cuda()
    output = model(random_data)
    print(model)