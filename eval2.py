import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

from utils import visualize_batch
import numpy as np
import matplotlib.image

from metrics.PyTorchEMD.emd import earth_mover_distance as EMD
from metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.ChamferDistancePytorch.fscore import fscore

from dataset.shapenet_data_pc import ShapeNet15kPointClouds

cham3D = chamfer_3DDist()

def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, accelerated_cd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    for sample_b_start in tqdm(iterator):
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr, _, _ = cham3D(sample_batch_exp.cuda(), ref_batch.cuda())
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1).detach().cpu())

            emd_batch = EMD(sample_batch_exp.cuda(), ref_batch.cuda(), transpose=False)
            emd_lst.append(emd_batch.view(1, -1).detach().cpu())

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd

def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }

def get_dataset(dataroot, npoints,category,use_mask=False):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True, use_mask = use_mask)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
        use_mask=use_mask
    )
    return tr_dataset, te_dataset


def main(config):
    logger = config.get_logger('eval')

    # Setup Data Loader for References
    _, test_dataset = get_dataset('data/ShapeNetCore.v2.PC15k/', 4096, 'car', use_mask=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, drop_last=False)
    
    ref = []
    for data in tqdm(test_dataloader, total=len(test_dataloader), desc='Generating Samples'):
        x = data['test_points']
        m, s = data['mean'].float(), data['std'].float()

        ref.append(x*s + m)

    ref_pcs = torch.cat(ref, dim=0).contiguous()
    print("ref:", ref_pcs.shape)

    # Setup Data Loader for Samples
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=16,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # Setup Model Architecture
    model = config.init_obj('arch', module_arch)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    #total_loss = 0.0
    #total_metrics = torch.zeros(len(metric_fns))


    

    with torch.no_grad():
        ref = []
        
        for i, data in enumerate(tqdm(data_loader)):
            print(len(data))
            print(data[0].shape)

            data = data.to(device)
            output = model(data)
            batch_size = data.shape[0]

            
            
            #x = data['test_points']
            #m, s = data['mean'].float(), data['std'].float()

            #ref.append(x*s + m)

        #ref_pcs = torch.cat(ref, dim=0).contiguous()

        # logger.info("Loading sample path: %s" % (opt.eval_path))
        # sample_pcs = torch.load(opt.eval_path).contiguous()

        # logger.info("Generation sample size:%s reference size: %s"
        #     % (sample_pcs.size(), ref_pcs.size()))


    # Compute metrics
            M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(data, ref_pcs, batch_size)

            res_cd = lgan_mmd_cov(M_rs_cd.t())
            print({
                "%s-CD" % k: v for k, v in res_cd.items()
            })

            res_emd = lgan_mmd_cov(M_rs_emd.t())
            print({
                "%s-EMD" % k: v for k, v in res_emd.items()
            })

    #n_samples = len(data_loader.sampler)
    #log = {'loss': total_loss / n_samples}
    #log.update({
    #    met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    #})
    #logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

