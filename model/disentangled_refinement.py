import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import knn
from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import grouping_operation


class SelfAttention(nn.Module):
    def __init__(self, in_dim, activation=None):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs:
                x : input feature maps (B X C X N)
            returns:
                out : self attention value + input feature 
                attention: B X N X N (N is the number of positions)
        """
        batch_size, C, N = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1) # B X N X C
        proj_key = self.key_conv(x).view(batch_size, -1, N) # B X C x N
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # B X N X N
        proj_value = self.value_conv(x).view(batch_size, -1, N) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B X C X N
        out = self.gamma * out + x
        return out

class GlobalRefinementUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalRefinementUnit, self).__init__()
        self.conv = nn.Conv1d(in_channels + 3, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.self_attention = SelfAttention(in_channels)

    def forward(self, F_E, Q_prime):
        """
        Args:
            F_E: Coarse feature map of shape (batch_size, C, rN)
            Q_prime: Coarse point cloud of shape (batch_size, 3, rN)
        
        Returns:
            Refined feature map of shape (batch_size, C, rN)
        """
        # Concatenate F_E and Q_prime
        combined = torch.cat([F_E, Q_prime], dim=1)  # (batch_size, C+3, rN)
        
        # Apply convolution, batch normalization and ReLU
        x = self.conv(combined)
        x = self.bn(x)
        x = self.relu(x)
        
        # Apply self-attention unit
        refined_features = self.self_attention(x)  # (batch_size, C, rN)
        
        return refined_features
    
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class LocalRefinementUnit(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(LocalRefinementUnit, self).__init__()
        self.k = k
        self.mlp1 = MLP(3, 64)  # Change according to the input feature dimension
        self.mlp2 = MLP(in_channels + 64, out_channels)
        self.mlp_w = MLP(3, k)  # For spatial weight regression
        
        self.knn_obj = KNN(k=k, transpose_mode=True)

    def forward(self, F_E, Q_prime):
        """
        Args:
            F_E: Coarse feature map of shape (batch_size, C, rN)
            Q_prime: Coarse point cloud of shape (batch_size, 3, rN)
        
        Returns:
            Refined local feature map of shape (batch_size, C, rN)
        """
        B, C, rN = F_E.size()
        _, _, K = Q_prime.size()
        
        print(F_E.shape)
        print(Q_prime.shape)

        # KNN grouping
        Q_prime = Q_prime.permute(0, 2, 1).contiguous()  # (batch_size, rN, 3)
        F_E = F_E.permute(0, 2, 1).contiguous()  # (batch_size, rN, C)

        #idx = knn(Q_prime, Q_prime, self.k)  # (batch_size * rN, k)
        
        _, idx = self.knn_obj(Q_prime, Q_prime)  # (batch_size, rN, k)
        # idx to int64
        idx = idx.int().contiguous()

        # Grouping Q'
        grouped_Q = grouping_operation(Q_prime.transpose(1, 2).contiguous(), idx) # (batch_size, 3, rN, k)
        grouped_Q = grouped_Q.permute(0, 2, 3, 1)  # (batch_size, rN, k, 3)

        # Grouping F_E
        grouped_F = grouping_operation(F_E.transpose(1, 2).contiguous(), idx) # (batch_size, 3, rN, k)
        grouped_F = grouped_F.permute(0, 2, 3, 1) # (batch_size, rN, k, C)
        #grouped_F = F_E.gather(1, idx_expanded)

        # Duplicate Q' with K copies
        Q_prime_dup = Q_prime.unsqueeze(2).expand(-1, -1, self.k, -1)  # (batch_size, rN, k, 3)

        # Subtraction for local encoding
        subtracted_Q = grouped_Q - Q_prime_dup  # (batch_size, rN, k, 3)

        # Encoding local feature volume
        subtracted_Q = subtracted_Q.view(B * rN * self.k, -1)  # (batch_size * rN * k, 3)
        encoded_sub_Q = self.mlp1(subtracted_Q)  # (batch_size * rN * k, 64)
        encoded_sub_Q = encoded_sub_Q.view(B, rN, self.k, -1)  # (batch_size, rN, k, 64)

        # Concatenation with grouped features
        concat_feature = torch.cat([grouped_F, encoded_sub_Q], dim=-1)  # (batch_size, rN, k, C + 64)
        concat_feature = concat_feature.view(B * rN * self.k, -1)  # (batch_size * rN * k, C + 64)
        encoded_feature = self.mlp2(concat_feature)  # (batch_size * rN * k, out_channels)
        encoded_feature = encoded_feature.view(B, rN, self.k, -1)  # (batch_size, rN, k, out_channels)

        # Spatial weight regression
        spatial_weight = self.mlp_w(subtracted_Q)  # (batch_size * rN * k, k)
        spatial_weight = spatial_weight.view(B, rN, self.k, self.k)  # (batch_size, rN, k, k)
        
        # Modify FL via convolution with W
        weighted_feature = torch.einsum('brkc,brkk->brkc', encoded_feature, spatial_weight)  # (batch_size, rN, k, out_channels)
        weighted_feature = weighted_feature.sum(dim=2)  # Summation along k-dimension

        # Final refined local features
        refined_local_features = weighted_feature + F_E  # Residual connection

        return refined_local_features.permute(0, 2, 1).contiguous()  # (batch_size, out_channels, rN)