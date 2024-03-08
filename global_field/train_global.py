# -*- coding: utf-8 -*-

import time
import torch

import torch.nn.functional as F
import argparse
import os
import numpy as np
import math

from tqdm import tqdm
from global_models.dataset import GlobalDataset
from global_models.fields import GlobalField
from pyhocon import ConfigFactory
from shutil import copyfile
from global_models.utils import get_root_logger, print_log, seed_all
from global_models.local import FPS, get_local_model

class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        self.conf = ConfigFactory.parse_file(conf_path)
        seed = self.conf.get_int('train.seed')
        
        if seed > 0:
            seed_all(seed)

        if args.dir != '':
            self.conf['general']['dir'] = args.dir
        self.all_file_dir = os.path.join(self.conf['general.base_exp_dir'], self.conf['general.dir'])
        os.makedirs(os.path.join(self.all_file_dir, '%dX'%(self.conf['dataset.up_rate'])), exist_ok=True)
        self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], self.conf['general.dir'], args.dataname)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset_global = GlobalDataset(self.conf['dataset'], args.dataname)
        self.dataname = args.dataname
        # Networks
        self.local_network = None
        self.load_local_checkpoint(args.dataset)
        self.global_field = GlobalField(**self.conf['model.global_field']).to(self.device)

        # Training parameters
        self.iter_step = 0
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.alpha = self.conf.get_float('train.alpha')
        self.beta_max = self.conf.get_float('train.beta_max')
        self.gamma = self.conf.get_float('train.gamma')
        self.mode = mode
        self.optimizer = torch.optim.AdamW(self.global_field.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        self.use_seed_upsample = self.conf.get_bool('train.use_seed_upsample', default=False)
        
        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        if os.path.exists(os.path.join(self.all_file_dir, self.dataset_global.up_name, self.dataname+'.xyz')):
            return  
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'global_{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size
        res_step = self.maxiter - self.iter_step
        self.local_network.eval()

        for iter_i in tqdm(range(res_step+1)):
            self.update_learning_rate()
            samples, nn = self.dataset_global.global_train_data(batch_size)
            samples.requires_grad = True

            # projecting the queries onto the field
            gradients_sample = self.global_field.gradient(samples).squeeze() # 5000x3
            udf_sample = self.global_field.udf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * udf_sample                 # 5000x3

            # computing loss
            sub_index = np.random.choice(batch_size, 100, replace=False)
            sub_sampled = sample_moved[sub_index, :].clone()
            knn, sub_sampled, fur = self.dataset_global.get_knn(sub_sampled, sub_sampled)
            dis = self.local_network(knn, sub_sampled)*fur
            local_loss = dis.mean()
            np_loss = torch.linalg.norm((sample_moved - nn), ord=2, dim=-1).mean()
            surf_loss = self.global_field(nn).mean()
            sp_loss = udf_sample.mean()
            global_loss = self.alpha * np_loss + self.beta_max * (self.maxiter-self.iter_step)/self.maxiter * surf_loss + self.gamma * sp_loss
            loss = local_loss + global_loss #+ 0.1 * abs_udf + 0.5* + npull_loss

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} all = {} local_loss = {} np_loss = {} surf_loss = {} sp_loss = {} lr={}'.format(self.iter_step, loss, local_loss, np_loss, surf_loss, sp_loss, self.optimizer.param_groups[0]['lr']), logger=logger)

            if self.iter_step % self.val_freq == 0:
                if not self.use_seed_upsample:
                    self.upsample(self.iter_step)
                else:
                    self.upsample_seed(self.iter_step)
                
            if self.iter_step % self.save_freq == 0: 
                self.save_checkpoint()    

            self.iter_step += 1

    def upsample(self, epoch=None):
        pointcloud = self.dataset_global.pointcloud.clone()
        surf_max = self.global_field.udf(pointcloud).max()
        up_rate = self.dataset_global.up_rate
        n_dis = self.dataset_global.nn_dis

        self.global_field.eval()
        results = []

        for i in range(0, pointcloud.shape[0], self.batch_size):
            l = min(self.batch_size, pointcloud.shape[0]-i)
            partial = pointcloud[i:i+l,:]
            part_dis = n_dis[i:i+l,:].unsqueeze(-1).repeat(1, up_rate*4, 3)
            noise = part_dis*(torch.rand(l, up_rate*4, 3)*2-1).to(self.device)
            query = torch.cat(((partial.unsqueeze(1) + noise).reshape(-1,3), partial), dim=0)  
            
            dis = self.global_field.udf(query)
            grad = self.global_field.gradient(query).squeeze()
            grad_norm = F.normalize(grad, dim=1)
            query = query - grad_norm*dis

            # filtering
            filter_udf = self.global_field(query).reshape(-1)
            query = query[filter_udf<=surf_max]
            results.append(query.detach().cpu().numpy())

        results = np.concatenate(results, axis=0).reshape(1,-1,3)
        results = torch.from_numpy(results).transpose(1,2).cuda()
        results = FPS(results, self.dataset_global.pointcloud.shape[0]*up_rate).transpose(1,2)[0].cpu()
        results = results*self.dataset_global.scale + self.dataset_global.centroid
        
        # (b, 3, fps_pts_num)
        if epoch is None:
            postfix = ''
        else:
            postfix = str(epoch)

        save_dir = os.path.join(self.base_exp_dir, 'outputs', self.dataset_global.up_name)

        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, self.dataname+'%s.xyz'%(postfix)), results)
        if epoch == self.maxiter:
            os.makedirs(os.path.join(self.all_file_dir, self.dataset_global.up_name), exist_ok=True)
            np.savetxt(os.path.join(self.all_file_dir, self.dataset_global.up_name, self.dataname+'.xyz'), results)
    
    def upsample_seed(self, epoch=None):
        from global_models.local import normalize_point_cloud
        from einops import rearrange
        pointcloud = self.dataset_global.pointcloud.clone()
        surf_max = self.global_field.udf(pointcloud).max()
        up_rate = self.dataset_global.up_rate
        samples = torch.from_numpy(np.loadtxt(os.path.join(self.conf['dataset']['data_dir'], "pussas_seed/%s.xyz.xyz"%(self.dataname))).astype(np.float32)[:,:3]).cuda()
        input_pcd = rearrange(samples, 'n c -> c n').contiguous()
        input_pcd = input_pcd.unsqueeze(0)
        samples, _, _ = normalize_point_cloud(input_pcd)
        samples = samples.squeeze().transpose(0,1)
        self.global_field.eval()  
        results = []

        for i in tqdm(range(0, samples.shape[0], self.batch_size)):
            l = min(self.batch_size, samples.shape[0]-i)
            query = samples[i:i+l]

            dis = self.global_field.udf(query)
            grad = self.global_field.gradient(query).squeeze()
            grad_norm = F.normalize(grad, dim=1)
            query = query - grad_norm*dis

            # filtering
            filter_udf = self.global_field(query).reshape(-1)
            query = query[filter_udf<=surf_max]
            results.append(query.detach().cpu().numpy())

        results = np.concatenate(results, axis=0).reshape(1,-1,3)
        results = torch.from_numpy(results).transpose(1,2).cuda()
        results = FPS(results, self.dataset_global.pointcloud.shape[0]*up_rate).transpose(1,2)[0].cpu()
        results = results*self.dataset_global.scale + self.dataset_global.centroid
        
        # (b, 3, fps_pts_num)
        if epoch is None:
            postfix = ''
        else:
            postfix = str(epoch)

        save_dir = os.path.join(self.base_exp_dir, 'outputs', self.dataset_global.up_name)
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, self.dataname+'%s.xyz'%(postfix)), results)
        if epoch == self.maxiter:
            os.makedirs(os.path.join(self.all_file_dir, self.dataset_global.up_name), exist_ok=True)
            np.savetxt(os.path.join(self.all_file_dir, self.dataset_global.up_name, self.dataname+'.xyz'), results)
    
    def update_learning_rate(self): 
        iter_step = self.iter_step
        max_iter = self.maxiter
        init_lr = self.learning_rate
        warn_up = self.warm_up_end
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_local_checkpoint(self, dataset):
        self.local_network = get_local_model(dataset).cuda()
        ckpt_path = self.conf['general.local_path']
        print("load local from %s"%(ckpt_path))
        self.local_network.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.local_network.eval()
        # avoid unnecessary backward
        self.local_network.set_global_mode()

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.global_field.load_state_dict(checkpoint['global_field'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        self.dataset_global.scale = checkpoint['scale']
        self.dataset_global.centroid = checkpoint['center']
        
    def save_checkpoint(self):
        checkpoint = {
            'global_field': self.global_field.state_dict(),
            'iter_step': self.iter_step,
            'optimizer': self.optimizer.state_dict(),
            'scale': self.dataset_global.scale,
            'center': self.dataset_global.centroid,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step))) 
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/pugan.conf')
    parser.add_argument('--mode', type=str, default='train', help='train or upsample')
    parser.add_argument('--dir', type=str, default='default_exp', help='exp dir')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataname', type=str, default='', help='global shape name')
    parser.add_argument('--listname', type=str, default='pugan.txt', help='list of all shapes in the dataset')
    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan for the pretrained local model')
    
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    if args.dataname == '':
        allfile = np.loadtxt(os.path.join("dataset_list", args.dataset+'.txt'), dtype=np.str_)
        print(allfile)
    else:
        allfile = [args.dataname]

    for name in allfile:
        args.dataname = name

        print("process shape", name)
        runner = Runner(args, args.conf, args.mode)

        if args.mode == 'train':
            runner.train()
        elif args.mode == 'upsample':
            epoch = 20000
            filename = os.path.join(runner.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(epoch))
            if os.path.exists(filename):
                runner.load_checkpoint('ckpt_{:0>6d}.pth'.format(epoch))
                '''Use the following code to match the reported results of pugan-4X and pu1k-4X datasets.
                The differences are caused by repeating calling torch.rand() in upsample() during training.
                The eleventh upsampled results are desired'''
                # for i in range(10):
                #     runner.upsample()
                if not runner.use_seed_upsample:
                    runner.upsample(epoch)
                else:
                    runner.upsample_seed(epoch)
            else:
                print("No such checkpoint!")
                print(filename)