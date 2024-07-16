import torch
import torch.nn.functional as F
from metrics.PyTorchEMD.emd import earth_mover_distance
from metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DFunction

# repulsion loss
from auction_match import auction_match
from pointnet2_ops import pointnet2_utils
#from pointnet2 import pointnet2_utils
from knn_cuda import KNN

from pytorch_fre.pytorch_fre_modules import FreLossPrecomputed, FreLoss
import math

from geomloss import SamplesLoss

# workaround to get pi
torch.pi = math.pi
#freLoss = FreLoss(512, 1024, 50, 50)
freLoss = None#FreLoss(256, 512, 50, 50)
geomEmdLoss = None

def fre_loss(output, target):
    global freLoss
    if freLoss is None:
        freLoss = FreLossPrecomputed(256, 512, 50, 50, k=3)
        #freLoss = FreLossPrecomputed(512, 1024, 50, 50, k=3)
    return freLoss(output, target)

def fre_loss2(output, target):
    global freLoss
    if freLoss is None:
        freLoss = FreLossPrecomputed(512, 1024, 50, 50, k=3, distance='spherical', s_knn=0.05)
    return freLoss(output, target)

def knn_point(group_size, point_cloud, query_cloud, transpose_mode=False):
    knn_obj = KNN(k=group_size, transpose_mode=transpose_mode)
    dist, idx = knn_obj(point_cloud, query_cloud)
    return dist, idx

def pu_emd_loss(output, target):
        idx, _ = auction_match(output, target)
        matched_out = pointnet2_utils.gather_operation(target.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (output - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,

        # we assume the point cloud is normalized to a unit sphere
        #dist2 /= pcd_radius
        return torch.mean(dist2)

# target gets ignored
def repulsion_loss(output, target):
    _, idx = knn_point(5, output, output, transpose_mode=True)
    idx = idx[:, :, 1:].to(torch.int32)  # remove first one
    idx = idx.contiguous()  # B, N, nn

    output = output.transpose(1, 2).contiguous()  # B, 3, N
    grouped_points = pointnet2_utils.grouping_operation(output, idx)  # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

    grouped_points = grouped_points - output.unsqueeze(-1)
    dist2 = torch.sum(grouped_points ** 2, dim=1)
    dist2 = torch.max(dist2, torch.tensor(1e-12).cuda())
    dist = torch.sqrt(dist2)
    weight = torch.exp(- dist2 / 0.03 ** 2)

    uniform_loss = torch.mean((0.07 - dist) * weight)
    # uniform_loss = torch.mean(self.radius - dist * weight) # punet
    return uniform_loss

def emd_re_loss(output, target):
    return 100 * pu_emd_loss(output, target) + repulsion_loss(output, target)
    #return 100 * emd_loss(output, target) + repulsion_loss(output, target)

#def cd_re_loss(output, target):
#    return cd_loss(output, target) + repulsion_loss(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def emd_loss(output, target):
    emd = earth_mover_distance(output, target, transpose=False)
    return torch.mean(emd)# / output.shape[1]

def cd_loss(output, target):
    loss = chamfer_3DFunction.apply(output, target)
    loss_for = loss[0]
    loss_bac = loss[1]
    return torch.mean(0.4 * loss_for + 0.6 * loss_bac)


def hausdorff_loss(P, Q):
        # P and Q are tensors of shape (batch_size, num_points, point_dim)
        # Compute the pairwise distance between points in P and Q
        P = P.unsqueeze(2)  # shape: (batch_size, num_points, 1, point_dim)
        Q = Q.unsqueeze(1)  # shape: (batch_size, 1, num_points, point_dim)
        
        # Compute pairwise distances
        distances = torch.norm(P - Q, dim=-1)  # shape: (batch_size, num_points, num_points)
        
        # Compute directed Hausdorff distance
        min_P_to_Q, _ = torch.min(distances, dim=2)  # shape: (batch_size, num_points)
        min_Q_to_P, _ = torch.min(distances, dim=1)  # shape: (batch_size, num_points)
        
        hausdorff_P_to_Q = torch.max(min_P_to_Q, dim=1)[0]  # shape: (batch_size)
        hausdorff_Q_to_P = torch.max(min_Q_to_P, dim=1)[0]  # shape: (batch_size)
        
        #hausdorff_distance = torch.max(hausdorff_P_to_Q, hausdorff_Q_to_P)
        hausdorff_distance = hausdorff_P_to_Q + hausdorff_Q_to_P
        
        return torch.mean(hausdorff_distance)
    
def geom_emd_loss(output, target):
    global geomEmdLoss
    if(geomEmdLoss is None):
        geomEmdLoss = SamplesLoss("sinkhorn", p=2, blur=0.05)
    return torch.mean(geomEmdLoss(output, target))
    