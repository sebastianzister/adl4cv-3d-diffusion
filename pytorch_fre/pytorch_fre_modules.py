import torch 
import torch.nn as nn
import torch_harmonics as th
import math

import pytorch_fre.pytorch_fre_utils as fre

import time

#DEBUG
import matplotlib.pyplot as plt

from knn_cuda import KNN

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
    phi = torch.arccos(coords[..., :-2]/phi_norms[..., :-2])
    phi_final = torch.arccos(coords[..., -2:-1]/phi_norms[..., -2:-1]) + (2*math.pi - 2*torch.arccos(coords[..., -2:-1]/phi_norms[..., -2:-1]))*(coords[..., -1:] < 0)
            
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

class FreCalc(nn.Module):
    def __init__(self, nlat=512, nlon=1024, lmax=50, mmax=50, device='cuda'):
        super(FreCalc, self).__init__()
        self.to(device)
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax
        self.mmax = mmax

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.nlat), torch.arange(-self.nlat, self.nlat))
        self.grid = torch.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).unsqueeze(0).to(device)
        self.grid = self.grid.float() / self.nlat * math.pi
        
        self.sht = th.RealSHT(self.nlat, self.nlon, grid='equiangular', lmax=self.lmax, mmax=self.mmax).to(device)
    
        self.knn_obj = KNN(k=3, transpose_mode=True)

    def forward(self, target):
        target_features, target_sph = to_spherical(target)
        target_sph[:, :, 1] -= math.pi
        
        #pred_dist, pred_idx = self.knn_obj(pred_sph, self.grid.expand(pred_sph.shape[0], -1, -1))
        target_dist, target_idx = self.knn_obj(target_sph, self.grid.expand(target_sph.shape[0], -1, -1))
        
        # OLD VERSION FOR NN
        #pred_dist, pred_idx = fre.three_nn(self.grid.expand(pred_sph.shape[0], -1, -1).contiguous(), pred_sph)
        #target_dist, target_idx = fre.three_nn(self.grid.expand(pred_sph.shape[0], -1, -1).contiguous(), target_sph)
        
        target_dist = target_dist/target_dist.sum(dim=-1, keepdim=True)
        
        target_interp = fre.three_interpolate(target_features.contiguous(), target_idx.int(), target_dist)
        
        target_coeffs = self.sht.forward(target_interp.reshape(-1, self.nlat, self.nlon))
        
        return target_coeffs.real
    

class FreLoss(nn.Module):
    def __init__(self, nlat=512, nlon=1024, lmax=50, mmax=50, device='cuda'):
        super(FreLoss, self).__init__()
        self.to(device)
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax
        self.mmax = mmax

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.nlat), torch.arange(-self.nlat, self.nlat))
        self.grid = torch.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).unsqueeze(0).to(device)
        self.grid = self.grid.float() / self.nlat * math.pi
        
        self.sht = th.RealSHT(self.nlat, self.nlon, grid='equiangular', lmax=self.lmax, mmax=self.mmax).to(device)
    
        self.s2_fre = self.lmax**2
        self.rect_weights = torch.exp(-((self.lmax - torch.arange(1, self.lmax+1))**2)/(2*self.s2_fre)).to(device)
        self.rect_weights = self.rect_weights.unsqueeze(0).unsqueeze(2)
        
        self.knn_obj = KNN(k=3, transpose_mode=True)

    def forward(self, pred, target):
        tmp_time = time.time()

        pred_features, pred_sph = to_spherical(pred)
        target_features, target_sph = to_spherical(target)
        pred_sph[:, :, 1] -= math.pi
        target_sph[:, :, 1] -= math.pi
        
        tmp_time = time.time()
        
        #pred_dist, pred_idx = self.knn_obj(pred_sph, self.grid.expand(pred_sph.shape[0], -1, -1))
        #target_dist, target_idx = self.knn_obj(target_sph, self.grid.expand(target_sph.shape[0], -1, -1))
        
        # OLD VERSION FOR NN
        pred_dist, pred_idx = fre.three_nn(self.grid.expand(pred_sph.shape[0], -1, -1).contiguous(), pred_sph)
        target_dist, target_idx = fre.three_nn(self.grid.expand(pred_sph.shape[0], -1, -1).contiguous(), target_sph)

        torch.cuda.synchronize()
        print('Time for NN: ', time.time() - tmp_time)

        
        tmp_time = time.time()

        
        pred_dist = pred_dist/pred_dist.sum(dim=-1, keepdim=True)
        target_dist = target_dist/target_dist.sum(dim=-1, keepdim=True)
        
        pred_interp = fre.three_interpolate(pred_features.contiguous(), pred_idx.int(), pred_dist)
        target_interp = fre.three_interpolate(target_features.contiguous(), target_idx.int(), target_dist)
        
        #plt.imshow(pred_interp[0].detach().cpu().numpy().reshape(self.nlat, self.nlon))
        #plt.show()

        torch.cuda.synchronize()
        print('Time for IP: ', time.time() - tmp_time)
        tmp_time = time.time()
        
        pred_coeffs = self.sht.forward(pred_interp.reshape(-1, self.nlat, self.nlon))
        target_coeffs = self.sht.forward(target_interp.reshape(-1, self.nlat, self.nlon))
        
        torch.cuda.synchronize()
        print('Time for SHT: ', time.time() - tmp_time)

        #return (pred_coeffs.real - target_coeffs.real)**2
        return torch.sum(((pred_coeffs.real - target_coeffs.real)**2)*self.rect_weights, dim=(1, 2)).mean()

class FreLossPrecomputed(nn.Module):
    def __init__(self, nlat=512, nlon=1024, lmax=50, mmax=50, device='cuda'):
        super(FreLossPrecomputed, self).__init__()
        self.to(device)
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax
        self.mmax = mmax

        grid_x, grid_y = torch.meshgrid(torch.arange(0, self.nlat), torch.arange(-self.nlat, self.nlat))
        self.grid = torch.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).unsqueeze(0).to(device)
        self.grid = self.grid.float() / self.nlat * math.pi
        
        self.sht = th.RealSHT(self.nlat, self.nlon, grid='equiangular', lmax=self.lmax, mmax=self.mmax).to(device)
    
        self.s2_fre = self.lmax**2
        self.rect_weights = torch.exp(-((self.lmax - torch.arange(1, self.lmax+1))**2)/(2*self.s2_fre)).to(device)
        self.rect_weights = self.rect_weights.unsqueeze(0).unsqueeze(2)
        
        self.knn_obj = KNN(k=3, transpose_mode=True)

    def forward(self, pred, target_coeffs):
        #tmp_time = time.time()

        pred_features, pred_sph = to_spherical(pred)
        pred_sph[:, :, 1] -= math.pi
        
        #tmp_time = time.time()
        
        pred_dist, pred_idx = self.knn_obj(pred_sph, self.grid.expand(pred_sph.shape[0], -1, -1))
        
        # OLD VERSION FOR NN
        #pred_dist, pred_idx = fre.three_nn(self.grid.expand(pred_sph.shape[0], -1, -1).contiguous(), pred_sph)

        #torch.cuda.synchronize()
        #print('Time for NN: ', time.time() - tmp_time)
        #tmp_time = time.time()

        
        pred_dist = pred_dist/pred_dist.sum(dim=-1, keepdim=True)
        
        pred_interp = fre.three_interpolate(pred_features.contiguous(), pred_idx.int(), pred_dist)
        
        #plt.imshow(pred_interp[0].detach().cpu().numpy().reshape(self.nlat, self.nlon))
        #plt.show()

        #torch.cuda.synchronize()
        #print('Time for IP: ', time.time() - tmp_time)
        #tmp_time = time.time()
        
        pred_coeffs = self.sht.forward(pred_interp.reshape(-1, self.nlat, self.nlon))
        
        #torch.cuda.synchronize()
        #print('Time for SHT: ', time.time() - tmp_time)

        #return (pred_coeffs.real - target_coeffs.real)**2
        return torch.sum(((pred_coeffs.real - target_coeffs)**2)*self.rect_weights, dim=(1, 2)).mean()

