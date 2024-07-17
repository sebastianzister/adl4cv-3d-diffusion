import torch 
import torch.nn as nn
import torch_harmonics as th
import math

#import pytorch_fre.pytorch_fre_utils as fre

import time

#DEBUG
import matplotlib.pyplot as plt

from knn_cuda import KNN

from pykeops.torch import LazyTensor, Genred

def to_spherical(coords: torch.Tensor) -> torch.Tensor:
    """
    Convert Cartesian coordinates to n-dimensional spherical coordinates.

    Args:
        coords (torch.Tensor): Tensor representing Cartesian coordinates (x_1, ... x_n).
                               Shape: (..., n)

    Returns:
        torch.Tensor: Tensor representing spherical coordinates (r, phi_1, ... phi_n-1).
                      Shape: (..., n)
    """    
    n = coords.shape[-1]
    
    # We compute the coordinates following https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    r = torch.norm(coords, dim=-1, keepdim=True)

    # phi_norms are the quotients in the wikipedia article above
    phi_norms = torch.norm(torch.tril(coords.flip(-1).unsqueeze(-2).expand((*coords.shape, n))), dim=-1).flip(-1)
    
    # replaced b/c of nan gradients
    #phi = torch.arccos(coords[..., :-2]/phi_norms[..., :-2])
    #phi_final = torch.arccos(coords[..., -2:-1]/phi_norms[..., -2:-1]) + (2*math.pi - 2*torch.arccos(coords[..., -2:-1]/phi_norms[..., -2:-1]))*(coords[..., -1:] < 0)

    eps = 1e-7

    # Clamp the input to arccos to be within (-1+eps, 1-eps)
    phi = torch.arccos((coords[..., :-2]/phi_norms[..., :-2]).clamp(-1+eps, 1-eps))
    phi_final = torch.arccos((coords[..., -2:-1]/phi_norms[..., -2:-1]).clamp(-1+eps, 1-eps)) + (2*math.pi - 2*torch.arccos((coords[..., -2:-1]/phi_norms[..., -2:-1]).clamp(-1+eps, 1-eps)))*(coords[..., -1:] < 0)

    #rho = torch.norm(coords, dim=-1, keepdim=True)
    #phi = torch.atan2(coords[..., 1:2], coords[..., 0:1])
    #theta = torch.acos(coords[..., 2:3] / rho)       
    #return rho.permute(0, 2, 1), torch.cat((phi, theta), dim=-1)
    return r.permute(0, 2, 1), torch.cat([phi, phi_final], dim=-1)

def to_cartesian(coords: torch.Tensor) -> torch.Tensor:
    """
    Convert n-dimensional spherical coordinates to Cartesian coordinates.

    Args:
        coords (torch.Tensor): Tensor representing spherical coordinates (r, phi_1, ... phi_n-1).
                               Shape: (..., n)

    Returns:
        torch.Tensor: Tensor representing Cartesian coordinates (x_1, ... x_n).
                      Shape: (..., n)
    """    
    n = coords.shape[-1]    
    r, phi = coords[..., 0:1], coords[..., 1:]
    
    phi_lower = torch.sin(torch.tril(phi.unsqueeze(-2).expand((*phi.shape, n-1))))
    phi_sin_prod = torch.prod(phi_lower + torch.triu(torch.ones((*phi.shape, n-1), device=coords.device), diagonal=1), dim=-1)
    
    x_1 = r * torch.cos(phi[..., 0:1])
    x_mid = r * torch.cos(phi[..., 1:]) * phi_sin_prod[..., :-1]
    x_n = r * phi_sin_prod[..., -1:]
    
    return torch.cat([x_1, x_mid, x_n], dim=-1)

# DEBUG REMOVE
class InterpolatePytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx, dist_scaled):
        # Get the number of batches, points, and neighbors
        bs, npoint, _ = features.shape
        _, ngrid, k = idx.shape
    
        neighbors = features.squeeze(1).expand(k, -1, -1).permute(1, 2, 0).gather(1, idx)
        interpolated_features = (neighbors * dist_scaled).sum(dim=2)
        
        ctx.save_for_backward(idx, dist_scaled, features)
        return interpolated_features
    @staticmethod
    def backward(ctx, grad_out):
        idx, dist_scaled, features = ctx.saved_tensors
        grad_features = grad_idx = grad_dist_scaled = None

        if ctx.needs_input_grad[0]:
            grad_features = torch.zeros_like(features)
            for i in range(grad_features.shape[0]):
                for j in range(grad_features.shape[1]):
                    grad_features[i, j, idx[i, j]] = grad_out[i, j] * dist_scaled[i, j]

        return grad_features, grad_idx, grad_dist_scaled

interpolate_pytorch = InterpolatePytorch.apply

class FreCalc(nn.Module):
    def __init__(self, nlat=512, nlon=1024, lmax=50, mmax=50, k=5, s_knn=None, distance='euclidean', device='cuda'):
        super(FreCalc, self).__init__()
        self.to(device)
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax
        self.mmax = mmax
        self.k = k

        if distance == 'spherical':
            if s_knn is None:
                print("WARNING: s_knn not provided, using default value of 0.05")
                s_knn = 0.05
            self.s2_knn = s_knn
            self.s2_fre = self.lmax
        elif distance == 'euclidean':
            self.s2_knn = s_knn**2 if s_knn is not None else None
            self.s2_fre = self.lmax**2
        else:
            raise ValueError('Distance must be either "spherical" or "euclidean"')
        self.distance = distance

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.nlat), torch.arange(-self.nlat, self.nlat))
        self.grid = torch.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).unsqueeze(0).to(device)
        self.grid = self.grid.float() / self.nlat * math.pi
        
        self.sht = th.RealSHT(self.nlat, self.nlon, grid='equiangular', lmax=self.lmax, mmax=self.mmax).to(device)
        
    
        #self.knn_obj = KNN(k=self.k, transpose_mode=True)

    def forward(self, target, visualize=False):
        target_features, target_sph = to_spherical(target)
        target_sph[:, :, 1] -= math.pi
        
        # PYKEOPS VERSION
        
        # SPHERICAL DISTANCE
        if(self.distance == 'spherical'):
            # Define the formula
            formula = "2 - 2 * (Sin(Elem(x_i, 0)) * Sin(Elem(y_j, 0)) * Cos(Elem(x_i, 1) - Elem(y_j, 1)) + Cos(Elem(x_i, 0)) * Cos(Elem(y_j, 0)))"
            
            # Define the aliases for the variables in the formula
            aliases = ["x_i = Vi(2)",  # x_i is a 2D vector per line
                       "y_j = Vj(2)"]  # y_j is a 2D vector per column
            
            # Create the Genred operation
            operation = Genred(formula, aliases, reduction_op='ArgKMin', axis=1, opt_arg=self.k)
            
            # Use the operation
            X_j = target_sph  # shape (B, 4096, 2)
            X_i = self.grid  # shape (B, 524288, 2)
            target_idx = operation(X_i, X_j)
    
            # Create a tensor of batch indices
            batch_indices = torch.arange(target_sph.shape[0]).view(-1, 1, 1)
    
            # Expand the batch indices tensor to match the shape of knn_idx
            batch_indices = batch_indices.expand(-1, self.grid.shape[1], self.k)
            # Use advanced indexing to select the k nearest neighbours
            target_tmp = target_sph[batch_indices, target_idx, :]
            # calculated the distances again
    
            ############
            # recalculate the spherical distance
            ############
            
            expanded_grid = self.grid.unsqueeze(-2).expand(target_sph.shape[0], -1, self.k, -1)
            target_dist = 2 - 2 * (torch.sin(target_tmp[..., 0])*torch.sin(expanded_grid[..., 0])*torch.cos(target_tmp[..., 1] - expanded_grid[..., 1]) + torch.cos(expanded_grid[..., 0])*torch.cos(target_tmp[..., 0]))

        elif(self.distance == 'euclidean'):
            X_j = LazyTensor(self.grid.expand(target_sph.shape[0], -1, -1).unsqueeze(-3))
            X_i = LazyTensor(target_sph.unsqueeze(-2))
            D2 = ((X_i - X_j) ** 2).sum(-1)
    
            target_idx = D2.argKmin(self.k, dim=1)
            # Create a tensor of batch indices
            batch_indices = torch.arange(target_sph.shape[0]).view(-1, 1, 1)
    
            # Expand the batch indices tensor to match the shape of knn_idx
            batch_indices = batch_indices.expand(-1, self.grid.shape[1], self.k)
            # Use advanced indexing to select the k nearest neighbours
            target_tmp = target_sph[batch_indices, target_idx, :]
            # calculated the distances again
            target_dist = ((target_tmp - self.grid.expand(target_sph.shape[0], -1, -1).unsqueeze(-2))**2).sum(-1)

        if(self.s2_knn is not None):
            target_dist = torch.exp(-target_dist/(2*self.s2_knn))


        # KNN_CUDA VERSION
        #target_dist, target_idx = self.knn_obj(target_sph, self.grid.expand(target_sph.shape[0], -1, -1))
        
        target_dist = target_dist/target_dist.sum(dim=-1, keepdim=True)
        
        # OLD VERSION FOR INTERPOLATE
        #target_interp = fre.three_interpolate(target_features.contiguous(), target_idx.int(), target_dist)

        target_interp = interpolate_pytorch(target_features.contiguous(), target_idx, target_dist).unsqueeze(1)
        
        target_coeffs = self.sht.forward(target_interp.reshape(-1, self.nlat, self.nlon))
        
        if visualize:
            return target_coeffs.real, (target_features, target_sph, target_interp)
        else:
            return target_coeffs.real
    
class FreLossPrecomputed(nn.Module):
    def __init__(self, nlat=512, nlon=1024, lmax=50, mmax=50, k=5, s_knn=None, distance='euclidean', device='cuda'):
        super(FreLossPrecomputed, self).__init__()
        self.to(device)
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax
        self.mmax = mmax
        self.k = k
        if distance == 'spherical':
            if s_knn is None:
                print("WARNING: s_knn not provided, using default value of 0.05")
                s_knn = 0.05
            self.s2_knn = s_knn
            self.s2_fre = self.lmax
        elif distance == 'euclidean':
            self.s2_knn = s_knn**2 if s_knn is not None else None
            self.s2_fre = self.lmax**2
        else:
            raise ValueError('Distance must be either "spherical" or "euclidean"')
        self.distance = distance

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.nlat), torch.arange(-self.nlat, self.nlat))
        self.grid = torch.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).unsqueeze(0).to(device)
        self.grid = self.grid.float() / self.nlat * math.pi
        
        self.sht = th.RealSHT(self.nlat, self.nlon, grid='equiangular', lmax=self.lmax, mmax=self.mmax).to(device)
    
        self.rect_weights = torch.exp(-((self.lmax - torch.arange(1, self.lmax+1))**2)/(2*self.s2_fre)).to(device)
        self.rect_weights = self.rect_weights.unsqueeze(0).unsqueeze(2)
        
        #self.knn_obj = KNN(k=3, transpose_mode=True)

    def forward(self, pred, target_coeffs):
        #tmp_time = time.time()
        
        if((target_coeffs == 0).all()):
            print("FreLoss::TARGET COEFFS ARE ZERO!")
            print("Did you forget to precaluclate the target coefficients?")

        pred_features, pred_sph = to_spherical(pred)
        pred_sph[:, :, 1] -= math.pi
        
        # SPHERICAL DISTANCE
        if self.distance == 'spherical':
            # Define the formula
            formula = "2 - 2 * (Sin(Elem(x_i, 0)) * Sin(Elem(y_j, 0)) * Cos(Elem(x_i, 1) - Elem(y_j, 1)) + Cos(Elem(x_i, 0)) * Cos(Elem(y_j, 0)))"
            
            # Define the aliases for the variables in the formula
            aliases = ["x_i = Vi(2)",  # x_i is a 2D vector per line
                       "y_j = Vj(2)"]  # y_j is a 2D vector per column
            
            # Create the Genred operation
            operation = Genred(formula, aliases, reduction_op='ArgKMin', axis=1, opt_arg=self.k)
            
            # Use the operation
            X_j = pred_sph  # shape (B, 4096, 2)
            X_i = self.grid  # shape (B, 524288, 2)
            pred_idx = operation(X_i, X_j)
    
            # Create a tensor of batch indices
            batch_indices = torch.arange(pred_sph.shape[0]).view(-1, 1, 1)
    
            # Expand the batch indices tensor to match the shape of knn_idx
            batch_indices = batch_indices.expand(-1, self.grid.shape[1], self.k)
            # Use advanced indexing to select the k nearest neighbours
            pred_tmp = pred_sph[batch_indices, pred_idx, :]
            # calculated the distances again
    
            ############
            # recalculate the spherical distance
            ############
            
            expanded_grid = self.grid.unsqueeze(-2).expand(pred_sph.shape[0], -1, self.k, -1)
            pred_dist = 2 - 2 * (torch.sin(pred_tmp[..., 0])*torch.sin(expanded_grid[..., 0])*torch.cos(pred_tmp[..., 1] - expanded_grid[..., 1]) + torch.cos(expanded_grid[..., 0])*torch.cos(pred_tmp[..., 0]))
        elif self.distance == 'euclidean':
            # PYKEOPS VERSION
            X_j = LazyTensor(self.grid.expand(pred_sph.shape[0], -1, -1).unsqueeze(-3).contiguous())
            X_i = LazyTensor(pred_sph.unsqueeze(-2))
            D2 = ((X_i - X_j) ** 2).sum(-1)
            pred_dist_sane, pred_idx = D2.Kmin_argKmin(self.k, dim=1)
    
            # Create a tensor of batch indices
            batch_indices = torch.arange(pred_sph.shape[0]).view(-1, 1, 1)
    
            # Expand the batch indices tensor to match the shape of knn_idx
            batch_indices = batch_indices.expand(-1, self.grid.shape[1], self.k)
            # Use advanced indexing to select the k nearest neighbours
            pred_tmp = pred_sph[batch_indices, pred_idx, :]
            # calculated the distances again usin pykeops
            
            pred_dist = ((pred_tmp - self.grid.expand(pred_sph.shape[0], -1, -1).unsqueeze(-2))**2).sum(-1)

        #pred_dist = torch.sqrt(pred_dist)
        if(self.s2_knn is not None):
            pred_dist_weighted = torch.exp(-pred_dist/(2*self.s2_knn))
        else:
            pred_dist_weighted = pred_dist

        # KNN_CUDA VERSION
        #pred_dist, pred_idx = self.knn_obj(pred_sph, self.grid.expand(pred_sph.shape[0], -1, -1))

        pred_dist_normalized = pred_dist_weighted/pred_dist_weighted.sum(dim=-1, keepdim=True)
        
        # OLD VERSION FOR INTERPOLATE
        #pred_interp = fre.three_interpolate(pred_features.contiguous(), pred_idx.int(), pred_dist_scaled)
        
        pred_interp = interpolate_pytorch(pred_features.contiguous(), pred_idx, pred_dist_normalized).unsqueeze(1)
        
        pred_coeffs = self.sht.forward(pred_interp.reshape(-1, self.nlat, self.nlon))
        
        dist_coeffs = ((pred_coeffs.real - target_coeffs)**2)

        dist_coeffs = dist_coeffs*self.rect_weights
        output = torch.sum(dist_coeffs, dim=(1, 2)).mean()

        return output

