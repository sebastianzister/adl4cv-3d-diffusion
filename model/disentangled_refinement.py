import torch
import torch.nn as nn
import torch.nn.functional as F

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