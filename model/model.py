import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import PointConvDensitySetAbstraction, visualize_point_cloud, visualize_multiple_point_clouds

from global_tags import GlobalTags
GlobalTags.legacy_layer_base(True)
from convpoint.nn.conv import PtConv
from convpoint.nn.utils import apply_bn

import MinkowskiEngine as ME

from model.pvcnn_generation import *

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SimpleModel(BaseModel):
    def __init__(self, num_points = 2048):
        super().__init__()
        self.fc1 = nn.Linear(num_points*3, num_points*3)
        
    def forward(self, pc, visualize_latent=False):
        B, P, C = pc.shape
        x = self.fc1(pc.flatten(start_dim=-2))
        return x.reshape(B, P, C)

class SharedSimpleModel(BaseModel):
    def __init__(self):
        super().__init__()
        hs = 256
        self.fc1 = nn.Linear(3, hs + 3)
        self.fc2 = nn.Linear(hs + 3, hs + 3)
        self.fc3 = nn.Linear(hs + 3, hs + 3)
        self.fc4 = nn.Linear(hs + 3, 3)
    def forward(self, pc, visualize_latent=False):
        x = F.relu(self.fc1(pc))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return pc + x

class PointDetailModel(BaseModel):
    def __init__(self, num_points = 2048):
        super().__init__()
        nsample = 5
        self.sa1 = PointConvDensitySetAbstraction(npoint=num_points, nsample=nsample, in_channel=3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=num_points//4, nsample=nsample, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.1, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=num_points//8, nsample=nsample, in_channel=256 + 3, mlp=[256, 256, 512], bandwidth = 0.1, group_all=False)
        self.isa1 = PointConvDensitySetAbstraction(npoint=num_points//8, nsample=nsample, in_channel=512 + 3, mlp=[512, 512, 256], bandwidth = 0.1, group_all=False)
        self.isa2 = PointConvDensitySetAbstraction(npoint=num_points//4, nsample=nsample, in_channel=256 + 3, mlp=[256, 256, 128], bandwidth = 0.1, group_all=False)
        self.isa3 = PointConvDensitySetAbstraction(npoint=num_points, nsample=nsample, in_channel=128 + 3, mlp=[128, 128, 64], bandwidth = 0.1, group_all=False)
        self.fc1 = nn.Linear(64 + 6, 64 + 6)
        self.fc2 = nn.Linear(64 + 6, 3)

        

    def forward(self, pc, visualize_latent=False):
        B, P, C = pc.shape
        pc = pc.permute(0, 2, 1)
        points1, features = self.sa1(pc, None)
        # do a max pooling of the features
        points2, features = self.sa2(points1, features)
        points3, features = self.sa3(points2, features)
        points1i, features = self.isa1(points3, features)
        points2i, features = self.isa2(points1i, features)
        points3i, features = self.isa3(points2i, features)

        points = torch.cat([points3i, features, pc], dim=1)
        points = self.fc1(points.permute(0, 2, 1))
        points = F.relu(points)
        points = self.fc2(points)
        print(points1i.shape)
        if(visualize_latent):
            visualize_multiple_point_clouds(
                [pc[0].permute(1, 0).cpu(), 
                points1[0].permute(1, 0).cpu(), 
                points2[0].permute(1, 0).cpu(), 
                points3[0].permute(1, 0).cpu(), 
                points1i[0].permute(1, 0).cpu(), 
                points2i[0].permute(1, 0).cpu(), 
                points3i[0].permute(1, 0).cpu(), 
                points.cpu()],
                ['Input', 'SA1', 'SA2', 'SA3', 'ISA1', 'ISA2', 'ISA3', 'Output']
            )


        #print(x.shape)
        # stack the features of the last layer with the input points
        
        #visualize_point_cloud(x[0].permute(1, 0).cpu())
        points3i.requires_grad = True

        return points3i.permute(0, 2, 1)

class ConvPointDetailModel(BaseModel):
    def __init__(self, input_channels=1, output_channels=3, dimension=3):
        super().__init__()
        n_centers = 16

        pl = 96
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4*pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)
        self.cv0d = PtConv(2*pl, pl, n_centers, dimension, use_bias=False)


        self.fcout = nn.Linear(pl, pl*2)
        self.fcout2 = nn.Linear(pl*2, pl*2)
        self.fcout3 = nn.Linear(pl*2, pl)
        self.fcout4 = nn.Linear(pl, output_channels)

        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2*pl)
        self.bn5 = nn.BatchNorm1d(2*pl)
        self.bn6 = nn.BatchNorm1d(2*pl)

        self.bn5d = nn.BatchNorm1d(2*pl)
        self.bn4d = nn.BatchNorm1d(2*pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(0.5)
    

    def forward(self, input_pts, return_features=False, visualize_latent=False):
    #def forward(self, x, input_pts, return_features=False):
        x = torch.ones(input_pts.size(0), input_pts.size(1), 1).to(input_pts.device)
        #normalize the input points between 0 and 1
        #input_pts = input_pts - input_pts.min()
        #input_pts = input_pts / input_pts.max()
        #print(input_pts.min(), input_pts.max())

        #print('Number of non zero values in: ', torch.count_nonzero(input_pts).item(), "/", input_pts.size(0)*input_pts.size(1)*input_pts.size(2))

        x1, pts1 = self.cv1(x, input_pts, 16, 2048)
        x1 = F.leaky_relu(apply_bn(x1, self.bn1))

        #print('Number of non zero values x1: ', torch.count_nonzero(x1).item(), "/", x1.size(0)*x1.size(1)*x1.size(2))

        x2, pts2 = self.cv2(x1, pts1, 16, 1024)
        x2 = F.leaky_relu(apply_bn(x2, self.bn2))

        #print('Number of non zero values x2: ', torch.count_nonzero(x2).item(), "/", x2.size(0)*x2.size(1)*x2.size(2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = F.leaky_relu(apply_bn(x3, self.bn3))

        #print('Number of non zero values x3: ', torch.count_nonzero(x3).item(), "/", x3.size(0)*x3.size(1)*x3.size(2))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = F.leaky_relu(apply_bn(x4, self.bn4))

        #print('Number of non zero values x4: ', torch.count_nonzero(x4).item(), "/", x4.size(0)*x4.size(1)*x4.size(2))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = F.leaky_relu(apply_bn(x5, self.bn5))

        #print('Number of non zero values x5: ', torch.count_nonzero(x5).item(), "/", x5.size(0)*x5.size(1)*x5.size(2))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = F.leaky_relu(apply_bn(x6, self.bn6))

        #print('Number of non zero values x6: ', torch.count_nonzero(x6).item(), "/", x6.size(0)*x6.size(1)*x6.size(2))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = F.leaky_relu(apply_bn(x5d, self.bn5d))
        #print('Number of non zero values x5d: ', torch.count_nonzero(x5d).item(), "/", x5d.size(0)*x5d.size(1)*x5d.size(2))
        x5d = torch.cat([x5d, x5], dim=2)

        
        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.leaky_relu(apply_bn(x4d, self.bn4d))
        #print('Number of non zero values x4d: ', torch.count_nonzero(x4d).item(), "/", x4d.size(0)*x4d.size(1)*x4d.size(2))
        x4d = torch.cat([x4d, x4], dim=2)
        

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = F.leaky_relu(apply_bn(x3d, self.bn3d))
        #print('Number of non zero values x3d: ', torch.count_nonzero(x3d).item(), "/", x3d.size(0)*x3d.size(1)*x3d.size(2))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = F.leaky_relu(apply_bn(x2d, self.bn2d))
        #print('Number of non zero values x2d: ', torch.count_nonzero(x2d).item(), "/", x2d.size(0)*x2d.size(1)*x2d.size(2))
        x2d = torch.cat([x2d, x2], dim=2)
        
        x1d, _ = self.cv1d(x2d, pts2, 8, pts1)
        x1d = F.leaky_relu(apply_bn(x1d, self.bn1d))
        #print('Number of non zero values x1d: ', torch.count_nonzero(x1d).item(), "/", x1d.size(0)*x1d.size(1)*x1d.size(2))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        x0d = F.leaky_relu(apply_bn(x0d, self.bn0d))

        xout = x0d
        #xout = xout.view(-1, xout.size(2))
        # max pool the features to get the global feature
        #xout_pool = torch.max(xout, dim=0)[0]
        # make the first dimension 2048
        #xout_pool = xout_pool.unsqueeze(0).repeat(xout.size(0), 1)
        ##print(xout_pool.shape)
        # append the global feature to the features
        #xout = torch.cat([xout_pool, xout], dim=1)
        #xout = torch.cat([xout, input_pts], dim = 2)
        # count and #print the number of non zero values
        #print('Number of non zero values ac: ', torch.count_nonzero(xout).item(), "/", xout.size(0)*xout.size(1)*xout.size(2))
        xout = F.leaky_relu(self.fcout(xout))
        # count and #print the number of non zero values
        #print('Number of non zero values 1: ', torch.count_nonzero(xout).item(), "/", xout.size(0)*xout.size(1)*xout.size(2))
        xout = F.leaky_relu(self.fcout2(xout))
        #print('Number of non zero values 2: ', torch.count_nonzero(xout).item(), "/", xout.size(0)*xout.size(1)*xout.size(2))
        xout = F.leaky_relu(self.fcout3(xout))
        #print('Number of non zero values 3: ', torch.count_nonzero(xout).item(), "/", xout.size(0)*xout.size(1)*xout.size(2))
        xout = self.fcout4(xout)
        #xout = xout.view(x.size(0), -1, xout.size(1))
        
        if visualize_latent:
            visualize_multiple_point_clouds(
                [input_pts[0].cpu(),
                pts1[0].cpu(),
                pts2[0].cpu(),
                pts3[0].cpu(),
                pts4[0].cpu(),
                pts5[0].cpu(),
                pts6[0].cpu(),
                xout[0].cpu(),
                xout[0].cpu() + input_pts[0].cpu(),
                ],
                ['Input', 'CV1', 'CV2', 'CV3', 'CV4', 'CV5', 'CV6', 'Output', 'Output + Input']
            )
            
            # plot the feature values

        print(xout.shape)

        if return_features:
            return xout, x0d
        else:
            return xout + input_pts

class SparseDetailModel(BaseModel, ME.MinkowskiNetwork):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self, in_channels=1, out_channels=3, dimension=3, D=3):
        BaseModel.__init__(self)
        ME.MinkowskiNetwork.__init__(self, D)

        # Input sparse tensor must have tensor stride 128.
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(1, enc_ch[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiELU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(
                enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiELU(),
        )

        # Decoder
        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6],
                dec_ch[5],
                kernel_size=4,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiELU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(
            dec_ch[5], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiELU(),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(
            dec_ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiELU(),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(
            dec_ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiELU(),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(
            dec_ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiELU(),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(
            dec_ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiELU(),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(
            dec_ch[0], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x):

        ##################################################
        # Convert the input tensor to a sparse tensor
        ##################################################
        feats = torch.ones(x.size(0), x.size(1), 1).to(x.device)
        coords = ME.utils.batched_coordinates([torch.floor(x[0]/0.01)]).to(x.device)
        x = ME.SparseTensor(feats[0], coords, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM)

        ##################################################
        # plot the sparse tensor using matplotlib
        ##################################################
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        ax.scatter(x.coordinates_at(0)[:, 0].cpu(), x.coordinates_at(0)[:, 1].cpu(), x.coordinates_at(0)[:, 2].cpu())
#        # set same aspect ratio
#        min_val = x.coordinates_at(0).min().cpu().item()
#        max_val = x.coordinates_at(0).max().cpu().item()
#        ax.set_xlim([min_val, max_val])
#        ax.set_ylim([min_val, max_val])
#        ax.set_zlim([min_val, max_val])
#        plt.show()


        ##################################################
        # Encoder
        ##################################################
        out_cls, targets = [], []

        enc_s1 = self.enc_block_s1(x)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        ##################################################
        # Decoder 64 -> 32
        ##################################################
        dec_s32 = self.dec_block_s64s32(enc_s64)

        # Add encoder features
        dec_s32 = dec_s32 + enc_s32
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        #target = self.get_target(dec_s32, target_key)
        #targets.append(target)
        out_cls.append(dec_s32_cls)

        #if self.training:
        #    keep_s32 += target

        # Remove voxels s32
        dec_s32 = self.pruning(dec_s32, keep_s32)

        ##################################################
        # Decoder 32 -> 16
        ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32)

        # Add encoder features
        dec_s16 = dec_s16 + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        #target = self.get_target(dec_s16, target_key)
        #targets.append(target)
        out_cls.append(dec_s16_cls)

        #if self.training:
        #    keep_s16 += target

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16)

        ##################################################
        # Decoder 16 -> 8
        ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16)

        # Add encoder features
        dec_s8 = dec_s8 + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)

        #target = self.get_target(dec_s8, target_key)
        #targets.append(target)
        out_cls.append(dec_s8_cls)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        #if self.training:
        #    keep_s8 += target

        # Remove voxels s16
        dec_s8 = self.pruning(dec_s8, keep_s8)

        ##################################################
        # Decoder 8 -> 4
        ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8)

        # Add encoder features
        dec_s4 = dec_s4 + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)

        #target = self.get_target(dec_s4, target_key)
        #targets.append(target)
        out_cls.append(dec_s4_cls)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        #if self.training:
        #    keep_s4 += target

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)

        ##################################################
        # Decoder 4 -> 2
        ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4)

        # Add encoder features
        dec_s2 = dec_s2 + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)

        #target = self.get_target(dec_s2, target_key)
        #targets.append(target)
        out_cls.append(dec_s2_cls)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        #if self.training:
        #    keep_s2 += target

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2)
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        # Add encoder features
        dec_s1 = dec_s1 + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        #target = self.get_target(dec_s1, target_key)
        #targets.append(target)
        out_cls.append(dec_s1_cls)
        keep_s1 = (dec_s1_cls.F > 0).squeeze()

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target

        # Remove voxels s1
        dec_s1 = self.pruning(dec_s1, keep_s1)
        out = dec_s1.C[:, 1:4].unsqueeze(0)*0.01
        print(out)
        return out

class PVCNN2Base(BaseModel):

    def __init__(self, num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)


        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, dropout, num_classes], # was 0.5
                                          classifier=False , dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)        

        t = torch.ones(inputs.shape[0]).to(inputs.device)

        temb =  self.embedf(self.get_timestep_embedding(t, inputs.device))[:,:,None].expand(-1,-1,inputs.shape[-1])

        # inputs : [B, in_channels + S, N]
        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for i, sa_blocks  in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks ((features, coords, temb))
            else:
                features, coords, temb = sa_blocks ((torch.cat([features,temb],dim=1), coords, temb))
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features)
        for fp_idx, fp_blocks  in enumerate(self.fp_layers):
            features, coords, temb = fp_blocks((coords_list[-1-fp_idx], coords, torch.cat([features,temb],dim=1), in_features_list[-1-fp_idx], temb))

        print(features.shape)

        return self.classifier(features).permute(0, 2, 1)        
    
class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes=3, embed_dim=1, use_att=False,dropout=0.0, extra_feature_channels=0, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        PVCNN2Base.__init__(self,
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )

        