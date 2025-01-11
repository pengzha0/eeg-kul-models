import torch.nn as nn
import torch
import math as m

class mapImage_mod(nn.Module):
    def __init__(self, input, imageSize: tuple = (16, 16), device='cpu') -> None:
        super().__init__()
        print("Initializing mapImage")
        print("Input size is")
        print(input.shape)
        self.imageSize = imageSize
        # batch * band * time * channel
        input = input.permute((0, 3, 1, 2)).unsqueeze(dim=1)
        # batch * any(1) * channel * band * time
        net = nn.Sequential(
            nn.Conv3d(
                in_channels=input.shape[1],
                out_channels=int(imageSize[0] * imageSize[1]),
                kernel_size=(input.shape[2], 1, 1),
                bias=False,
                padding='valid',
            ),
            # nn.ReLU(),
            nn.BatchNorm3d(int(imageSize[0] * imageSize[1])),
        ).to(device)
        input = net(input).squeeze()
        self.add_module("mapnet", net)
        # batch * band * time * imagesize

        input = input.view(
            input.shape[0],
            imageSize[0],
            imageSize[1],
            input.shape[-2],
            input.shape[-1],
        )
        input = input.permute((0, 3, 4, 1, 2))
        # batcg * band * time * c1 * c2
        print("Output size is")
        print(input.shape)

    def forward(self, input):
        input = input.permute((0, 3, 1, 2)).unsqueeze(dim=1)
        input = self.mapnet(input).squeeze()
        input = input.view(
            input.shape[0],
            self.imageSize[0],
            self.imageSize[1],
            input.shape[-2],
            input.shape[-1],
        )
        input = input.permute((0, 3, 4, 1, 2))
        return input
    
class channelAttention(nn.Module):
    def __init__(
        self, input, kernelSize=(1, 3, 3), imageSize=(8, 8,), device='cpu'
    ) -> None:
        super().__init__()
        # batch * band * frame * channel
        print("Initializing channel attention")
        print("Input size is")
        print(input.shape)
        # input = networkChMap(input)
        net = mapImage_mod(input, imageSize, device=device)
        input = net(input)
        self.add_module("map", net)
        # batch * band *frame * c1*c2
        input = input.permute((0, 2, 1, 3, 4))
        # batch * frame * band * c1 * c2
        a = nn.AvgPool3d((input.shape[2], 1, 1)).to(device)
        m = nn.MaxPool3d((input.shape[2], 1, 1)).to(device)
        Fa = a(input)
        Fm = m(input)
        # feature: batch * time * 1 * c1 * c2
        f = torch.concat((Fa, Fm), 2)
        # feature: batch * time * 2 * c1 * c2
        f = f.permute((0, 2, 1, 3, 4))
        # feature : batch * 2 * time * c1 * c2
        # f = f.permute((0, 3, 1, 2))
        convnet = nn.Sequential(
            nn.Conv3d(
                f.shape[1], 1, (1, kernelSize[1], kernelSize[2]), padding="same",
            ),
            nn.ELU(),
            nn.BatchNorm3d(1),
            nn.Dropout(),
            nn.AvgPool3d((f.shape[2], 1, 1)),
        ).to(device)
        # feature: batch * 1 * 1 * c1 * c2
        f = torch.sigmoid(convnet(f))
        input = input.permute((0, 2, 1, 3, 4)) * f
        # input : batch * band * frame * c1 * c2
        self.add_module("avgp", a)
        self.add_module("maxp", m)
        self.add_module('conv', convnet)
        print("Output size is")
        print(input.shape)
        # bf * band * c1 * c2

    def forward(self, input):
        # batch * band * frame * channel
        # input = networkChMap(input)
        input = self.map(input)
        # batch * band *frame * c1*c2
        input = input.permute((0, 2, 1, 3, 4))
        # batch * frame * band * c1 * c2
        Fa = self.avgp(input)
        Fm = self.maxp(input)
        # feature: batch * time * 1 * c1 * c2
        f = torch.concat((Fa, Fm), 2)
        # feature: batch * time * 2 * c1 * c2
        f = f.permute((0, 2, 1, 3, 4))
        # feature * batch * 2 * time * c1 * c2
        f = torch.sigmoid(self.conv(f))
        # feature: batch * 1 * time * c1 * c2
        input = input.permute((0, 2, 1, 3, 4)) * f
        # input : batch * band * frame * c1 * c2
        return input
    

class bandAttention(nn.Module):
    def __init__(self, input, reduction, device='cpu') -> None:
        super().__init__()
        # batch * band * frame * channel
        avgp = nn.AvgPool2d((input.shape[-2], input.shape[-1])).to(device)
        maxp = nn.MaxPool2d((input.shape[-2], input.shape[-1])).to(device)
        Favg = avgp(input).squeeze()
        Fmax = maxp(input).squeeze()
        self.add_module("avgp", avgp)
        self.add_module("maxp", maxp)
        fcnet = nn.Sequential(
            nn.Linear(input.shape[1], input.shape[1] // reduction),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(input.shape[1] // reduction, input.shape[1]),
            nn.Tanh(),
            nn.Dropout(),
        ).to(device)
        self.add_module("fcnet", fcnet)
        Favg = fcnet(Favg)
        Fmax = fcnet(Fmax)
        Fmask = torch.sigmoid(Favg + Fmax)
        input = input * Fmask.unsqueeze(dim=-1).unsqueeze(dim=-1)

    def forward(self, input):
        M = torch.sigmoid(
            self.fcnet(self.avgp(input).squeeze())
            + self.fcnet(self.maxp(input).squeeze())
        )
        input = input * M.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return input

class deepNetwork(nn.Module):
    """Some Information about deepNetwork"""

    def __init__(
        self,
        input,
        numConvFilter,
        numFCNeurons,
        numGroup=None,
        imageSize=(9, 11),
        imageKernelSize=3,
        device='cuda',
    ):
        super(deepNetwork, self).__init__()

        self.numconv = len(numConvFilter)
        self.numfc = len(numFCNeurons)
        if numGroup is None:
            self.numGroup = numFCNeurons[-1]
        else:
            self.numGroup = numGroup

        input = input.to(device)
        print(input.shape)
        self.freqAttention = bandAttention(input, 1, device=device).to(device)
        input = self.freqAttention(input)
        # batch * band * frame * channel
        self.channelAttention = channelAttention(
            input, imageKernelSize, imageSize, device=device
        ).to(device)
        input = self.channelAttention(input)

        self.convnet = nn.ModuleList()
        maxpoolIdx = 0
        print(input.shape)
        # (batch * frame ) * band * c1 * c2
        for convLayerIdx in range(self.numconv):
            # (Batch * Time) * Band * Chan1 * Chan2
            inputChannel = input.shape[1]
            # convolutionKernel = (inputHeight, inputWidth)
            convolutionKernel = tuple(
                [m.ceil(kernel / (2 ** maxpoolIdx)) for kernel in imageKernelSize]
            )
            conv = nn.Conv3d(
                inputChannel,
                numConvFilter[convLayerIdx],
                convolutionKernel,
                padding='same',
            )
            relu = nn.ReLU()
            bn = nn.BatchNorm3d(numConvFilter[convLayerIdx])
            dp = nn.Dropout()
            net = nn.Sequential(conv, relu, bn, dp)
            net = net.to(device)
            input = net(input)
            self.convnet.add_module("conv%02d" % (convLayerIdx + 1), net)
            print(input.shape)

        # Average pooling
        avgpKernel1 = input.shape[2]
        avgpKernel2 = 1
        avgpKernel3 = 1
        net = nn.AvgPool3d((avgpKernel1, avgpKernel2, avgpKernel3))
        self.add_module("avgp", net)
        net = net.to(device)
        input = net(input)
        # Average pooling across time dimension
        print(input.shape)

        net = nn.Flatten()
        net = net.to(device)
        input = net(input)
        self.add_module("channelFlatten", net)
        print(input.shape)

        # Fully Connect
        input = input.squeeze()
        self.fcnet = nn.ModuleList()
        for fcLayerIdx in range(self.numfc):
            inputChannel = input.shape[1]
            outputChannel = numFCNeurons[fcLayerIdx]
            fc = nn.Linear(inputChannel, outputChannel)
            # dp = nn.Dropout()
            net = nn.Sequential(fc, dp)
            net = net.to(device)
            input = net(input)
            self.fcnet.add_module("fc%02d" % (fcLayerIdx + 1), net)
            print(input.shape)


    def forward(self, input):
        input = self.freqAttention(input)
        input = self.channelAttention(input)
        for model in iter(self.convnet):
            input = model(input)

        # Average pooling
        # Batch * Time * Feature * Chan1' * Chan2'
        input = self.avgp(input)
        input = self.channelFlatten(input)
        # Fully Connect
        input = input.squeeze()
        for model in iter(self.fcnet):
            input = model(input)

        return input

