import functools

import torch.nn as nn
import torch
import numpy as np
from modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule, Attention, Swish


def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8,out_channels), Swish())


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1, activations=None, use_bn=True):
    '''
    Create the shared mlp components.

    in_channels: int, the number of input channels
    out_channels: int or list of int, the number of output channels
    classifier: bool, whether to use the classifier
    dim: int, the dimension of the input
    width_multiplier: float, the width multiplier

    return: a tuple of three elements:
        - layers: list of nn.Module, the shared mlp components
        - out_channels: int, the number of output channels
        - in_channels: int, the number of input channels
    '''
    r = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc, dim=dim))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1]), activations=activations, use_bn=use_bn, dim=dim))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, embed_dim, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    c = 0
    for k, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
        out_channels = int(r * out_channels)
        for p in range(num_blocks):
            attention = k % 2 == 0 and k > 0 and p == 0
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                          with_se=with_se, normalize=normalize, eps=eps)

            if c == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(in_channels+embed_dim, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
            c += 1
    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                   dropout=0.1, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
    '''
    Create the shared mlp and pointnet modules of the pointnet++ architecture.

    sa_blocks: list of tuples, each tuple contains two elements:
        - the first element is a tuple of three elements:
            - out_channels: int, the number of output channels of the shared mlp
            - num_blocks: int, the number of shared mlp blocks
            - voxel_resolution: int, the resolution of the voxel grid
        - the second element is a tuple of four elements:
            - num_centers: int, the number of centers
            - radius: float, the radius of the ball query
            - num_neighbors: int, the number of neighbors
            - out_channels: int, the number of output channels of the pointnet module
    extra_feature_channels: int, the number of channels of the extra features
    embed_dim: int, the dimension of the embedding
    use_att: bool, whether to use attention
    dropout: float, the dropout rate
    with_se: bool, whether to use squeeze-and-excitation
    normalize: bool, whether to use batch normalization
    eps: float, the epsilon of batch normalization
    width_multiplier: float, the width multiplier
    voxel_resolution_multiplier: float, the voxel resolution multiplier

    return: a tuple of four elements:
        - sa_layers: list of nn.Module, the shared mlp and pointnet modules
        - sa_in_channels: list of int, the number of input channels of the shared mlp and pointnet modules
        - in_channels: int, the number of output channels of the shared mlp and pointnet modules
        - num_centers: int, the number of centers
    '''
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    for conv_configs, sa_configs in sa_blocks:
        sa_in_channels.append(in_channels)
        sa_blocks = []
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                          with_se=with_se, normalize=normalize, eps=eps)
            for _ in range(num_blocks):
                sa_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
            extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_blocks.append(block(in_channels=extra_feature_channels, out_channels=out_channels,
                               include_coordinates=True))
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, embed_dim=64, use_att=False,
                                dropout=0.1,
                                with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    '''
    Create the pointnet++ feature propagation modules.

    fp_blocks: list of tuples, each tuple contains two elements:
        - the first element is a tuple of two elements:
            - out_channels: int, the number of output channels of the pointnet module
            - num_blocks: int, the number of pointnet blocks
        - the second element is a tuple of three elements:
            - out_channels: int, the number of output channels of the shared mlp
            - num_blocks: int, the number of shared mlp blocks
            - voxel_resolution: int, the resolution of the voxel grid
    in_channels: int, the number of input channels of the feature propagation modules
    sa_in_channels: list of int, the number of input channels of the shared mlp and pointnet modules
    embed_dim: int, the dimension of the embedding
    use_att: bool, whether to use attention
    dropout: float, the dropout rate
    with_se: bool, whether to use squeeze-and-excitation
    normalize: bool, whether to use batch normalization
    eps: float, the epsilon of batch normalization
    width_multiplier: float, the width multiplier
    voxel_resolution_multiplier: float, the voxel resolution multiplier

    return: a tuple of two elements:
        - fp_layers: list of nn.Module, the feature propagation modules
        - in_channels: int, the number of output channels of the feature propagation modules
    '''

    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim, out_channels=out_channels)
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c+1) % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
                                              dropout=dropout,
                                              with_se=with_se, with_se_relu=True,
                                              normalize=normalize, eps=eps)

                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        c += 1

    return fp_layers, in_channels






