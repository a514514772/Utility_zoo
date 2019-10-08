import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from tqdm import tqdm
from torchvision import transforms
'''
import dataset
'''
# from dataset import CelebADataset

'''
import needed utilities from other python files
'''
# from ops import VGGLoss
# from utils import mkdirs, grid2gif, Logger
# from model import AutoEncoder, Discriminator, AttributeClassifier


class Solver(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.data_root = args.data_root
        self.lr = args.lr
        self.max_iter = args.max_iter
        self.num_workers = args.num_workers
        self.pbar = tqdm(total=self.max_iter)

        self.img_dir = args.img_dir
        mkdirs(self.img_dir)
        self.log_dir = args.log_dir
        self.logger = Logger(self.log_dir)
        self.global_iter = 0
        self.ckpt_dir = args.ckpt_dir
        mkdirs(self.ckpt_dir)

        # Other parameters

        # model initialization

        # optims

        # losses

        # dataset initialization
        trans = transforms.Compose([transforms.CenterCrop((178, 178)),
                                    transforms.Resize(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.pbar.write('preparing data')
        self.dset = CelebADataset(self.data_root, transforms=trans, num=50000)
        self.loader = DataLoader(self.dset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.pbar.write('finished preparing')

    def train(self):
        stop = False
        while not stop:
            for imgs, labels in self.loader:
                self.global_iter += 1
                self.pbar.update(1)
                '''
                    Adjusting learning rates if needed
                '''
                #self.adjust_learning_rate([self.opt_cclf, self.opt_iclf], self.lr*10, self.global_iter,
                #                          max_iter=self.max_iter * 2, power=0.9)

                imgs = imgs.cuda()
                labels = labels.cuda().float()

                '''
                 training procedure
                '''

                # Loggers and saving checkpoints
                '''
                if self.global_iter % 100 == 0:
                    self.logger.update('prob_fake2', prob_fake2.mean().item())
                if self.global_iter % 1000 == 0:
                    self.dump_image(rec, str(self.global_iter))
                    self.logger.plot_curve()
                if self.global_iter % 10000 == 0:
                    self.save_checkpoint(ckptname=str(self.global_iter))
                '''
                # stop criterion
                if self.global_iter >= self.max_iter:
                    stop = True

    def eval(self):
        print (self.ae)
        self.load_checkpoint(ckptname=str(21000))
        self.latent_traverse(key='test')

    def switch_mode(self, train):
        assert type(train) == bool

        if train:
            for k, v in self.nets.items():
                v.train()
        else:
            for k, v in self.nets.items():
                v.eval()

    def switch_disc(self, switch_flag):
        assert type(switch_flag) == bool

        for key in self.discs:
            for p in self.discs[key].parameters():
                p.require_grads = switch_flag

    def dump_image(self, img, name, nrow=8, img_size=(128, 128), normalize=True):
        if (img.size(2), img.size(3)) != img_size:
            img = F.interpolate(img, size=img_size)
        grid_img = make_grid(img, nrow=nrow, normalize=normalize, range=(-1, 1))
        save_image(grid_img, os.path.join(self.img_dir, name + '.png'))

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {}
        optim_states = {}
        for key, value in self.nets.items():
            model_states[key] = value.state_dict()
        for key, value in self.opts.items():
            optim_states[key] = value.state_dict()

        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.pbar.update(self.global_iter)
            for key, value in self.nets.items():
                value.load_state_dict(checkpoint['model_states'][key])
            for key, value in self.opts.items():
                value.load_state_dict(checkpoint['optim_states'][key])

            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))

    def poly_lr_scheduler(self, base_lr, iter, max_iter=30000, power=0.9):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def adjust_learning_rate(self, opts, base_lr, i_iter, max_iter, power):
        lr = self.poly_lr_scheduler(base_lr, i_iter, max_iter, power)
        for opt in opts:
            opt.param_groups[0]['lr'] = lr


if __name__ == '__main__':
    interpolation = torch.arange(-1, 1, 0.2)
    a = torch.arange()
