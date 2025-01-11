import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, channels_num, lstm_hidden_size=128, lstm_layers=3, output_size=2):
        super(CNN_LSTM, self).__init__()
        
        # CNN部分：只有一个卷积层
        self.conv1 = nn.Conv1d(channels_num, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.pooling1 = nn.MaxPool1d(2)  # 可选的池化层，用于下采样

        # LSTM部分：两层LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(channels_num, output_size)
        
        # 激活函数
        self.act = nn.PReLU()

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 输入维度 (batch_size, 通道数, 时间特征)
        x = x.permute(0, 2, 1)
        # CNN部分
        x = self.conv1(x)  # 卷积操作
        x = self.batchnorm1(x)  # 批归一化
        x = self.act(x)  # 激活函数
        x = self.pooling1(x)  # 池化下采样，减小时间特征长度
        
        # 调整维度，准备输入LSTM
        # x = x.permute(0, 2, 1)  # (batch_size, 时间特征, 通道数)，转换为LSTM需要的形状

        # LSTM部分：两层LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out形状为 (batch_size, 时间特征, lstm_hidden_size)

        # 取最后一个时间步的输出作为特征
        lstm_out = lstm_out.mean(-1)  # (batch_size, lstm_hidden_size)
        
        # 全连接层
        out = self.fc(lstm_out)  # 输出分类结果
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='tanh')
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


if __name__ == '__main__':
    channels_num = 50  # 假设有50个通道
    model = EEGNet_CNN_LSTM(channels_num).cuda()
    data = torch.zeros((16, channels_num, 4560)).cuda()  # 假设输入数据的形状是 (batch_size, 通道数, 时间特征)
    out = model(data)
    print(out.shape)  # 输出形状应为 (batch_size, 输出类别数)
