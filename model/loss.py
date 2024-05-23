import torch.nn.functional as F
#from metrics.PyTorchEMD.emd import earth_mover_distance
from metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DFunction


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

#def emd_loss(output, target):
#    return earth_mover_distance(output, target, transpose=False)

def cd_loss(output, target):
    out = chamfer_3DFunction.apply(output, target)
    return out[0].mean()
