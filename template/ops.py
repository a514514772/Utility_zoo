"""ops.py"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class HLoss(nn.Module):
    def __init__(self):
        self.sigmoid = torch.nn.Sigmoid()
        self.log_sigmoid = torch.nn.LogSigmoid()
        super(HLoss, self).__init__()

    def forward(self, x):
        b = self.sigmoid(x) * self.log_sigmoid(x)
        b = -1.0 * b.sum(1)
        return b.mean()
        
def kernel_density_ratio(logits, reduction='mean'):
    # output: E[log(D(x)/1-D(x))] = E[bce(D(x, true)-bce(D(x), False)]
    pt = F.binary_cross_entropy_with_logits(logits, torch.ones(logits.size()).cuda(), reduction='none')
    pf = F.binary_cross_entropy_with_logits(logits, torch.zeros(logits.size()).cuda(), reduction='none')

    assert reduction in ['mean', 'sum', 'none'], 'unsupported reduction type %s' % reduction

    output = None
    if reduction == 'mean':
        output = (pt-pf).mean()
    elif reduction == 'sum':
        output = (pt-pf).sum()
    elif reduction == 'none':
        output = pt-pf
    return output

def recon_loss(x_recon, x):
    n = x.size(0)
    loss = F.mse_loss(x_recon, x, reduction='none').div(n)
    return loss.sum()


def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
    return kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        #self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        bs = x.size(0)
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


if __name__ == '__main__':
    output = kernel_density_ratio(torch.ones(5)*-10000, reduction='none')
    print(output)