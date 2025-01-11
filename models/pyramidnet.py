import torch
import torch.nn as nn
import math
# from math import round
import torch.utils.model_zoo as model_zoo


def conv3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = conv3(planes, planes)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, (planes * 1), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d((planes * 1))
        self.conv3 = nn.Conv1d((planes * 1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm1d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                                       featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


class PyramidNet(nn.Module):

    def __init__(self, depth=32, alpha=300, num_classes=10, bottleneck=False):
        super(PyramidNet, self).__init__()
        # self.dataset = dataset

        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                  200: [3, 24, 36, 3]}

        if layers.get(depth) is None:
            if bottleneck == True:
                blocks[depth] = Bottleneck
                temp_cfg = int((depth - 2) / 12)
            else:
                blocks[depth] = BasicBlock
                temp_cfg = int((depth - 2) / 8)

            layers[depth] = [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
            print('=> the layer configuration for each stage is set to', layers[depth])

        self.inplanes = 64
        self.addrate = alpha / (sum(layers[depth]) * 1.0)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv1d(64, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.input_featuremap_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.featuremap_dim = self.input_featuremap_dim
        self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
        self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
        self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
        self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = nn.BatchNorm1d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:  # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = nn.AvgPool1d(2, stride=2, ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(
                block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)# torch.Size([16, 64, 13])

        x = self.layer1(x)
        # print(x.shape)# torch.Size([16, 139, 13])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)# torch.Size([16, 364, 2])

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def PYRAMIDNET(channels_num, sample_len):
    return PyramidNet(depth=18,num_classes=2)

if __name__ == '__main__':
    model = PyramidNet().cuda()
    data = torch.zeros((16, 50, 4560)).cuda()
    out = model(data)
    print(out.shape)  # torch.Size([16, 20])







