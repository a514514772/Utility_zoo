"""model.py"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn import utils

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False),
            #nn.BatchNorm2d(dim_out, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=False))
            #nn.BatchNorm2d(dim_out, affine=True, track_running_stats=False))

    def forward(self, x):
        return x + self.main(x)


class ConvBlcok(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, norm='none', activation='none'):
        super(ConvBlcok, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)

        self.norm = None
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=False)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_dim, affine=True, track_running_stats=True)

        self.act_fn = None
        if activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.conv(x)
        if self.norm is not None:
            output = self.norm(output)
        if self.act_fn is not None:
            output = self.act_fn(output)
        return output

class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, norm='none', activation='none'):
        super(DeconvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)

        self.norm = None
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_dim, affine=True, track_running_stats=False)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_dim, affine=True, track_running_stats=True)


        self.act_fn = None
        if activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.conv(x)
        if self.norm is not None:
            output = self.norm(output)
        if self.act_fn is not None:
            output = self.act_fn(output)
        return output


class AttributeClassifier(nn.Module):
    def __init__(self, cp_dim, cp_ext):
        super(AttributeClassifier, self).__init__()
        self.cp_dim = cp_dim
        self.cp_ext = cp_ext
        self.clf = nn.Sequential(
            nn.Linear(cp_ext, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, c_p):
        reshaped_z = c_p.reshape(-1, self.cp_ext)
        pred = self.clf(reshaped_z)

        return pred.view(-1, self.cp_dim)


class ImageClassifier(nn.Module):
    def __init__(self, cp_dim):
        super(ImageClassifier, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.fc = nn.Sequential(nn.Linear(64*4*4, 100),
        nn.LeakyReLU(0.2),
        nn.Linear(100, cp_dim))

    def forward(self, img):
        output = self.disc(img)
        output = self.fc(torch.flatten(output, 1))
        return output


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=4, repeat_num=4):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(utils.spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(utils.spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src.squeeze(), out_cls.squeeze()


class AutoEncoder(nn.Module):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""
    def __init__(self, cp_dim, cu_dim):
        super(AutoEncoder, self).__init__()
        self.cp_dim = cp_dim
        self.cu_dim = cu_dim
        self.z_dim = cp_dim + cu_dim
        self.encoder = nn.Sequential(
            ConvBlcok(3, 64, 3, 2, 1, activation='relu', norm='instance'),
            ConvBlcok(64, 64, 3, 1, 1, activation='relu', norm='instance'),
            ConvBlcok(64, 128, 3, 2, 1, activation='relu', norm='instance'),
            ConvBlcok(128, 128, 3, 1, 1, activation='relu', norm='instance'),
            ConvBlcok(128, 256, 3, 2, 1, activation='relu', norm='instance'),
            ConvBlcok(256, 256, 3, 1, 1, activation='relu', norm='instance'),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        self.enc_cp = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    ConvBlcok(256, self.cp_dim, 1, 1, 0))
        self.enc_cu = nn.Sequential(ConvBlcok(256, self.cu_dim, 3, 1, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(self.z_dim, 256, 1),
            nn.ReLU(True),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            DeconvBlock(256, 128, 4, 2, 1, activation='relu', norm='instance'),
            ConvBlcok(128, 128, 3, 1, 1, activation='relu', norm='instance'),
            DeconvBlock(128, 64, 4, 2, 1, activation='relu', norm='instance'),
            ConvBlcok(64, 64, 3, 1, 1, activation='relu', norm='instance'),
            DeconvBlock(64, 32, 4, 2, 1, activation='relu', norm='instance'),
            ConvBlcok(32, 32, 3, 1, 1, activation='relu', norm='instance'),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def decode(self, c_p, c_u):
        c_p = c_p.view(c_p.size(0), c_p.size(1), 1, 1).expand(c_p.size(0), c_p.size(1), c_u.size(2), c_u.size(3))
        z = torch.cat((c_p, c_u), 1)
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x, no_dec=False):
        feat = self.encoder(x)
        c_p = self.enc_cp(feat)
        c_u = self.enc_cu(feat)

        x_recon = self.decode(c_p, c_u)
        return x_recon, c_p.squeeze(), c_u

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
