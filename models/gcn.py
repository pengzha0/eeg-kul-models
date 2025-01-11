import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyChannelAttention(nn.Module):
    def __init__(self, channels_num):
        super(MyChannelAttention, self).__init__()
        # Global average pooling followed by two fully connected layers
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(2)  # Global Average Pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(channels_num, 4)  # First linear layer
        self.fc2 = nn.Linear(4, channels_num)  # Second linear layer
        self.tanh = nn.Tanh()  # Activation function

    def forward(self, inputs):
        inputs = inputs.permute(0, 1, 3, 2)
        # Global average pooling: output shape will be (batch_size, channels, 1, 1)
        x = torch.mean(inputs, dim=2)
        
        # Reshape: (batch_size, channels, 1, 1) -> (batch_size, channels)
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers with activation
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = x.mean(0)
        return x

class MyGraphConvolution(nn.Module):
    def __init__(self, channels_num, graph_convolution_kernel, is_channel_attention):
        super(MyGraphConvolution, self).__init__()
        
        self.channels_num = channels_num
        # Load adjacency matrix
        adjacency = np.zeros((64, 64))  # Example adjacency matrix
        edges = np.load('/home/kul/models/edges.npy')
        for x, y in edges:
            adjacency[x][y] = 1
            adjacency[y][x] = 1
        adjacency = np.sign(adjacency + np.eye(channels_num))
        adjacency = np.sum(adjacency, axis=0) * np.eye(64) - adjacency
        
        # Eigen decomposition of the adjacency matrix
        e_values, e_vectors = np.linalg.eig(adjacency)
        self.e_vectors = torch.tensor(e_vectors, dtype=torch.float32)

        # Define the graph kernel as a learnable parameter
        self.graph_kernel_param = nn.Parameter(torch.randn(graph_convolution_kernel, 1, channels_num))

        # Add channel attention mechanism if needed
        self.graph_channel_attention = MyChannelAttention(channels_num) if is_channel_attention else None

    def forward(self, x):

        graph_kernel_param = self.graph_kernel_param.to(x.device)
        # Precompute the graph kernel in the constructor
        kernel = graph_kernel_param * torch.eye(self.channels_num).to(x.device)
        graph_kernel = self.e_vectors.to(x.device) @ kernel.squeeze() @ self.e_vectors.T.to(x.device)
        graph_kernel = graph_kernel.unsqueeze(0)  # Expand dimension to match input

        # Use precomputed graph_kernel
        graph_kernel = graph_kernel.to(x.device)

        # Apply channel attention if needed
        if self.graph_channel_attention is not None:
            cha_attention = self.graph_channel_attention(x)
            graph_kernel = cha_attention * graph_kernel

        # Apply graph convolution
        x = torch.matmul(graph_kernel, x)
        x = F.relu(x)

        return x


class GCN(nn.Module):
    def __init__(self, channels_num, sample_len, graph_layer_num, graph_convolution_kernel, is_channel_attention):
        super(GCN, self).__init__()

        # Batch normalization layer
        self.batch_norm1 = nn.BatchNorm2d(1)

        # Graph convolution layers and batch norm layers
        self.graph_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(graph_layer_num):
            self.graph_convs.append(MyGraphConvolution(channels_num, graph_convolution_kernel, is_channel_attention))
            self.batch_norms.append(nn.BatchNorm2d(graph_convolution_kernel))

        # Fully connected layers
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(channels_num*graph_convolution_kernel, 100)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 2)
        
    def forward(self, x):
        # Permute (batch, time, channel) to (batch, channel, time)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)  # Reshape to (batch, 1, channels, time)

        # Apply initial batch normalization
        x = self.batch_norm1(x)

        # Apply graph convolution and batch normalization layers
        for conv, bn in zip(self.graph_convs, self.batch_norms):
            x = conv(x)
            x = bn(x)

        # Permute to match the shape (batch, channels, height, width)
        x = x.permute(0, 1, 3, 2)

        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, x.shape[-1]))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout and activations
        x = self.dropout1(x)
        x = F.tanh(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        # Softmax for classification
        x = F.softmax(x, dim=1)

        # x = torch.rand(128, 2).to(device)


        return x














# import torch
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# import torch.nn.functional as F
# import numpy as np

# import torch_geometric.nn as pyg_nn
# from torch_geometric.utils import from_scipy_sparse_matrix
# import scipy.sparse as sp

# class GCN(torch.nn.Module):
#     """
#     The graph convolutional operator from the "Semi-supervised
#     Classification with Graph Convolutional Networks"
#     <https://arxiv.org/abs/1609.09207>
#     """

#     def __init__(self, channels_num=64, sample_len=128,):
#         super(GCN, self).__init__()
#         # 增加更多的图卷积层
#         self.conv1 = GCNConv(sample_len, 128)
#         self.conv2 = GCNConv(128, 256)
#         self.conv3 = GCNConv(256, 512)
#         self.conv4 = GCNConv(512, 256)
#         self.conv5 = GCNConv(256, 128)
#         self.conv6 = GCNConv(128, 64)

#         self.num_nodes = channels_num

#         # 加载邻接矩阵
#         adj_matrix = np.load('/home/kul/models/edges.npy')  # (324, 2) 表示时间步之间的连接关系
#         self.edge_index = torch.tensor(adj_matrix.T, dtype=torch.long).contiguous()

#         # 更新全连接层
#         self.fc = torch.nn.Linear(64*2, 2)

#     def append(self, edge_index, batch_size):  # stretch and repeat and rename
#         edge_index_all = torch.LongTensor(2, edge_index.shape[1] * batch_size)
#         data_batch = torch.LongTensor(self.num_nodes * batch_size)
#         for i in range((batch_size)):
#             edge_index_all[:, i*edge_index.shape[1]:(i+1)*edge_index.shape[1]] = edge_index + i * self.num_nodes
#             data_batch[i*self.num_nodes:(i+1)*self.num_nodes] = i
#         return edge_index_all.to(self.device), data_batch.to(self.device)
    
#     def forward(self, data):
#         x = data  # 输入的data: (batch_size, channel, time_features)

#         x = x.permute(0,2,1)
        
#         batch_size, channel, time_features = x.size()
#         self.device = data.device
#         edge_index, data_batch = self.append(self.edge_index, batch_size)
#         # 将输入数据转换为图的特征
#         x = x.reshape(-1, x.shape[-1])
#         # self.edge_index = self.edge_index.to(data.device)
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.relu(self.conv3(x, edge_index))
#         x = F.relu(self.conv4(x, edge_index))
#         x = F.relu(self.conv5(x, edge_index))
#         x = F.relu(self.conv6(x, edge_index))

#         x = torch.cat([gmp(x, data_batch), gap(x, data_batch)], dim=1)
#         # x = x.mean(dim=1)
#         x = self.fc(x)
        
#         return x