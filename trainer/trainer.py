import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, visualize_batch, visualize_multiple_point_clouds

from torch.profiler import profile, record_function, ProfilerActivity

import time

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, criterion_weight, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        # added for multiple losses
        self.criterion_weight = criterion_weight

        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # start timer
        start_time = time.time()

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # changed to multiple losses
            loss = 0
            for i, c in enumerate(self.criterion):
                loss += self.criterion_weight[i] * c(output, target)
            loss.backward()
            self.optimizer.step()
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
        
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            # visualize first train batch
            if self.config['trainer']['visualize_train_batch'] is not None and epoch % self.config['trainer']['visualize_train_batch'] == 0 and batch_idx == 0:
                self.writer.add_image('train_batch', visualize_batch(data.cpu().detach(), output.cpu().detach(), target.cpu().detach()))
        
            if batch_idx == self.len_epoch:
                break
                
            #print(prof.key_averages().table(sort_by="cuda_time_total"))
            #print(prof.key_averages().table(sort_by="cpu_time_total"))
        log = self.train_metrics.result()

        if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.config['trainer']['visualize_per_epochs'] is not None and epoch % self.config['trainer']['visualize_per_epochs'] == 0:
            self._visualize_val_batch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        # end timer
        print("Epoch {} took {:.2f} seconds".format(epoch, time.time() - start_time))
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                
                #visualize_multiple_point_clouds([data[0].cpu(), output[0].cpu(), target[0].cpu()], ['Input', 'Output', 'Target'])
                
                loss = 0
                for i, c in enumerate(self.criterion):
                    loss += self.criterion_weight[i] * c(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
#        for name, p in self.model.named_parameters():
#            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
    
    def _visualize_val_batch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            batch_idx, (data, target) = next(enumerate(self.valid_data_loader))
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data, visualize_latent=True)
            
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'visualize')
            self.writer.add_image('target_result', visualize_batch(data.cpu(), output.cpu(), target.cpu()))


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
