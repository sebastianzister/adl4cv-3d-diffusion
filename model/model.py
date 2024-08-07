
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# PVCNN2
from model.pvcnn_generation import *

# PU-Net
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
import utils.punet_module as pt_utils

# Visualizations
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

# PVCNN2 orig
#from model.pvcnn_utils import *

# disentagled refinement
from model.disentangled_refinement import GlobalRefinementUnit, LocalRefinementUnit

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
        self.fc1 = nn.Linear(3, hs)
        self.fc2 = nn.Linear(hs, 2*hs)
        self.fc3 = nn.Linear(2*hs, hs)
        self.fc4 = nn.Linear(hs, 3)
    def forward(self, pc, visualize_latent=False):
        x = F.relu(self.fc1(pc))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

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

        # add 3 in_channels for the xyz
        layers, _ = create_mlp_components(in_channels=channels_fp_features+3, out_channels=[128, dropout, num_classes], # was 0.5
                                          classifier=False , dim=2, width_multiplier=width_multiplier, activations=[None], use_bn=False)
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

    def forward(self, inputs, visualize_latent=False):
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

        features = torch.cat([features, coords], dim=1).unsqueeze(2)
        #print(features.shape)

        return self.classifier(features).squeeze(2).permute(0, 2, 1)      
    
class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (2048, 0.05, 32, (32, 64))),
        ((64, 3, 16), (1024, 0.1, 32, (64, 128))),
        ((128, 3, 8), (512, 0.2, 32, (128, 256))),
        (None, (256, 0.3, 32, (256, 256, 512))),
    ]

    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes=3, embed_dim=64, use_att=False,dropout=0.0, extra_feature_channels=0, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        PVCNN2Base.__init__(self,
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )

# ---------------------------------------------------------------------------------------
class PUNet(BaseModel):
    def __init__(self, npoint=1024, up_ratio=4, use_normal=False, use_bn=False, use_global_ref=False):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio
        
        self.use_global_ref = use_global_ref

        self.npoints = [
            npoint, 
            npoint // 2, 
            npoint // 4, 
            npoint // 8
        ]

        mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]

        radius = [0.05, 0.1, 0.2, 0.3]

        nsamples = [32, 32, 32, 32]

        # for 4 downsample layers
        in_ch = 0 if not use_normal else 3
        self.SA_modules = nn.ModuleList()
        for k in range(len(self.npoints)):
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[k],
                    radius=radius[k],
                    nsample=nsamples[k],
                    mlp=[in_ch] + mlps[k],
                    use_xyz=True,
                    bn=use_bn))
            in_ch = mlps[k][-1]

        # upsamples for layer 2 ~ 4
        self.FP_Modules = nn.ModuleList()
        for k in range(len(self.npoints) - 1):
            self.FP_Modules.append(
                PointnetFPModule(
                    mlp=[mlps[k + 1][-1], 64], 
                    bn=use_bn))
        
        # feature Expansion
        in_ch = len(self.npoints) * 64 + 3 # 4 layers + input xyz
        self.FC_Modules = nn.ModuleList()
        for k in range(up_ratio):
            self.FC_Modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 256, 128],
                    bn=use_bn))

        # coordinate reconstruction
        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False)) 
        
        if(use_global_ref):
            self.global_ref = GlobalRefinementUnit(128, 128)
        
            self.pcd_layer2 = nn.Sequential(
                pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
                pt_utils.SharedMLP([64, 3], activation=None, bn=False))

    def forward(self, points, npoint=None, visualize_latent=False, return_features=False):
        if npoint is None:
            npoints = [None] * len(self.npoints)
        else:
            npoints = []
            for k in range(len(self.npoints)):
                npoints.append(npoint // 2 ** k)

        # points: bs, N, 3/6
        xyz = points[..., :3].contiguous()
        feats = points[..., 3:].transpose(1, 2).contiguous() if self.use_normal else None


        # downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            lk_xyz, lk_feats = self.SA_modules[k](l_xyz[k], l_feats[k])
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        # upsample
        up_feats = []
        for k in range(len(self.FP_Modules)):
            upk_feats = self.FP_Modules[k](xyz, l_xyz[k + 2], None, l_feats[k + 2])
            up_feats.append(upk_feats)

        # aggregation
        # [xyz, l0, l1, l2, l3]
        feats = torch.cat([
            xyz.transpose(1, 2).contiguous(),
            l_feats[1],
            *up_feats], dim=1).unsqueeze(-1)  # bs, mid_ch, N, 1

        # expansion
        r_feats = []
        for k in range(len(self.FC_Modules)):
            feat_k = self.FC_Modules[k](feats) # bs, mid_ch, N, 1
            r_feats.append(feat_k)
        r_feats = torch.cat(r_feats, dim=2) # bs, mid_ch, r * N, 1

        # reconstruction
        recon = self.pcd_layer(r_feats)  # bs, 3, r * N, 1
        if(self.use_global_ref):
            ref_feats = self.global_ref(r_feats.squeeze(-1), recon.squeeze(-1))
            output = self.pcd_layer2(ref_feats.unsqueeze(-1))
        else:
            output = recon
            
        if(return_features):
            return output.squeeze(-1).contiguous(), r_feats.squeeze(-1)
        return output.squeeze(-1).transpose(1, 2).contiguous() # bs, 3, r * N

# ---------------------------------------------------------------------------------------
# PUNet -> PVCU
# ---------------------------------------------------------------------------------------
class PVCU(BaseModel):
    def __init__(self, npoint=1024, up_ratio=4, use_normal=False, use_bn=False):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio

        self.npoints = [
            npoint, 
            npoint // 2, 
        ]

        mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]

        radius = [0.05, 0.1, 0.2, 0.3]

        nsamples = [32, 32, 32, 32]

        # for 4 downsample layers
        in_ch = 0 if not use_normal else 3

        sa_blocks = [
            ((32, 1, 128), (2048, 0.05, 32, (32, 32, 64))),
            ((64, 1, 64), (1024, 0.1, 32, (64, 64, 128))),
        ]
        
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=sa_blocks, extra_feature_channels=0, with_se=False, embed_dim=0,
            use_att=False, dropout=0.0, width_multiplier=1, voxel_resolution_multiplier=1
        )

        self.SA_modules = nn.ModuleList(sa_layers)


        # upsamples for layer 2 ~ 4
        fp_blocks = [
            ((64, 64), (64, 1, 128)),
        ]

        channels_sa_features = 0
        sa_in_channels = [0, 128] #[0, 512-64, 256-64, 128]

        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=False, 
            embed_dim=0, use_att=False, dropout=0.0, width_multiplier=1, voxel_resolution_multiplier=1
        )
        self.FP_Modules = nn.ModuleList(fp_layers)

        # feature Expansion
        in_ch = len(self.npoints) * 64 + 3 # 4 layers + input xyz
        self.FC_Modules = nn.ModuleList()
        for k in range(up_ratio):
            self.FC_Modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 256, 128],
                    bn=use_bn))

        # coordinate reconstruction
        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False)) 

    def forward(self, points, npoint=None, visualize_latent=False):
        if npoint is None:
            npoints = [None] * len(self.npoints)
        else:
            npoints = []
            for k in range(len(self.npoints)):
                npoints.append(npoint // 2 ** k)

        points = points.permute(0, 2, 1)
        # inputs : [B, in_channels + S, N]
        xyz = points[:, :3, :].contiguous()
        feats = points[:, :, :].contiguous()
        
        # downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            lk_feats, lk_xyz = self.SA_modules[k]((l_feats[k], l_xyz[k]))
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        # upsample
        up_feats = []
        for k in range(len(self.FP_Modules)):
            points_coords = xyz
            points_features = None
            centers_coords = l_xyz[k + 2]
            centers_features = l_feats[k + 2] #.permute(1, 0, 2)

            # order should be: centers_coords, points_coords, centers_features (according to pvd, but then 1024 points are missing)
            upk_feats, upk_coords = self.FP_Modules[k]((points_coords, centers_coords, centers_features))
            up_feats.append(upk_feats) #.permute(1, 0, 2))

        feats = torch.cat([
            xyz,
            l_feats[1], #should be l_feats[1] but dimensions are wrong
            *up_feats
        ], dim=1).unsqueeze(-1)  # bs, mid_ch, N, 1

        # expansion
        r_feats = []
        for k in range(len(self.FC_Modules)):
            feat_k = self.FC_Modules[k](feats) # bs, mid_ch, N, 1
            r_feats.append(feat_k)
        r_feats = torch.cat(r_feats, dim=2) # bs, mid_ch, r * N, 1

        # reconstruction
        output = self.pcd_layer(r_feats)  # bs, 3, r * N, 1
        return output.squeeze(-1).transpose(1, 2).contiguous() # bs, 3, r * N

class PVCU_Fast(BaseModel):
    def __init__(self, npoint=1024, up_ratio=4, use_normal=False, use_bn=False):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio

        self.npoints = [
            npoint, 
            npoint // 2, 
        ]

        mlps = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]

        radius = [0.05, 0.1, 0.2, 0.3]

        nsamples = [32, 32, 32, 32]

        # for 4 downsample layers
        in_ch = 0 if not use_normal else 3

        sa_blocks = [
            ((32, 1, 32), (2048, 0.05, 32, (32, 32, 64))),
            ((64, 1, 16), (1024, 0.1, 32, (64, 64, 128))),
        ]
        
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=sa_blocks, extra_feature_channels=0, with_se=False, embed_dim=0,
            use_att=False, dropout=0.0, width_multiplier=1, voxel_resolution_multiplier=1
        )

        self.SA_modules = nn.ModuleList(sa_layers)

        # upsamples for layer 2 ~ 4
        fp_blocks = [
            ((64, 64), (64, 1, 32)),
#            ((64, 64), (64, 1, 32)),
#            ((64, 64), (64, 1, 16)),
        ]

        channels_sa_features = 0
        sa_in_channels = [0, 256-64, 128] #[0, 512-64, 256-64, 128]

        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=False, 
            embed_dim=0, use_att=False, dropout=0.0, width_multiplier=1, voxel_resolution_multiplier=1
        )
        self.FP_Modules = nn.ModuleList(fp_layers)

        # feature Expansion
        in_ch = len(self.npoints) * 64 + 3 # 4 layers + input xyz
        self.FC_Modules = nn.ModuleList()
        for k in range(up_ratio):
            self.FC_Modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 256, 128],
                    bn=use_bn))

        # coordinate reconstruction
        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False)) 

    def forward(self, points, npoint=None, visualize_latent=False):
        if npoint is None:
            npoints = [None] * len(self.npoints)
        else:
            npoints = []
            for k in range(len(self.npoints)):
                npoints.append(npoint // 2 ** k)

        points = points.permute(0, 2, 1)
        # inputs : [B, in_channels + S, N]
        xyz = points[:, :3, :].contiguous()
        feats = points[:, :, :].contiguous()

        # downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            lk_feats, lk_xyz = self.SA_modules[k]((l_feats[k], l_xyz[k]))
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        # upsample
        up_feats = []
        for k in range(len(self.FP_Modules)):
            points_coords = xyz
            points_features = None
            centers_coords = l_xyz[k + 2]
            centers_features = l_feats[k + 2] #.permute(1, 0, 2)
            
            # order should be: centers_coords, points_coords, centers_features (according to pvd, but then 1024 points are missing)
            upk_feats, upk_coords = self.FP_Modules[k]((points_coords, centers_coords, centers_features))
            up_feats.append(upk_feats) #.permute(1, 0, 2))

        feats = torch.cat([
            xyz,
            l_feats[1],
            *up_feats
        ], dim=1).unsqueeze(-1)  # bs, mid_ch, N, 1

        # expansion
        r_feats = []
        for k in range(len(self.FC_Modules)):
            feat_k = self.FC_Modules[k](feats) # bs, mid_ch, N, 1
            r_feats.append(feat_k)
        r_feats = torch.cat(r_feats, dim=2) # bs, mid_ch, r * N, 1

        # reconstruction
        output = self.pcd_layer(r_feats)  # bs, 3, r * N, 1
        return output.squeeze(-1).transpose(1, 2).contiguous() # bs, 3, r * N
        
class PUNetRef(BaseModel):
    def __init__(self, npoint=1024, up_ratio=4, use_normal=False, use_bn=False, checkpoint_path=None):
        super().__init__()
        
        self.expand = PUNet(npoint, up_ratio, use_normal, use_bn)
        

        

        if(checkpoint_path is not None):
            self.expand.load_state_dict(torch.load(checkpoint_path)['state_dict'])
            for param in self.expand.parameters():
                param.requires_grad = False
        
        self.global_ref = GlobalRefinementUnit(128, 128)
        
        self.local_ref = LocalRefinementUnit(128, 128, 32)
        
        in_ch = 128
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 3], activation=None, bn=False)) 
        
    def forward(self, points, visualize_latent=False):
        coarse_xyz, coarse_feat = self.expand(points, return_features=True)
        
        global_ref_feats = self.global_ref(coarse_feat, coarse_xyz)
        local_ref_feats = self.local_ref(coarse_feat, coarse_xyz)
        
        fine_feats = global_ref_feats + local_ref_feats
        
        offset = self.pcd_layer(fine_feats.unsqueeze(-1))
        output = offset.squeeze(-1)#coarse_xyz + offset.squeeze(-1)
        return output.transpose(1, 2).contiguous()

