import math
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image

from utils import save_models

import sys
old_stdout = sys.stdout
sys.stdout = open('stdout.txt', 'w')

DATA_PATH = '/home/u5397696/data/svhn'
num_classes = 10
max_epochs = 1000

batch_size = 64
train_t = transforms.Compose([transforms.Resize(32),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),),])

test_t = transforms.Compose([transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = datasets.SVHN(DATA_PATH, download=False,transform=train_t)
print (len(train_set))

test_set = datasets.SVHN(DATA_PATH, download=False,transform=test_t, split='test')

indices = np.random.permutation(73257)
print (len(test_set))

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(range(52231)))

val_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=2,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(range(52231, len(train_set))))

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    num_workers=2,
    shuffle=False)

print (len(val_loader), len(test_loader))

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, sobel=True):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096)
        )
        
        self.relu = nn.ReLU()
        
        self.top_layer = nn.Linear(4096, num_classes)
        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.relu(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                #print(y)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None):
        self.name = name
        self.log_list = None
        self.reset()

    def reset(self):
        if self.log_list is not None:
            self.log_list.append(self.avg)
        else:
            self.log_list = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count

clf = VGG('VGG16', num_classes).cuda()
opt_clf = optim.SGD(clf.parameters(), lr=1e-3, momentum=0.9, weight_decay=10e-5)

ce_crit = nn.CrossEntropyLoss().cuda()

model_dict = {}
model_dict['clf'] = clf

max_acc = -1e9
for epoch in range(max_epochs):
    pred_losses = AverageMeter()
    val_acc = AverageMeter()
    test_acc = AverageMeter()

    clf.train()
    for it, (img, label) in enumerate(train_loader):
        imgv = Variable(img).cuda()
        labelv = Variable(label).cuda()
        
        pred = clf(imgv)
        pred_loss = ce_crit(pred, labelv)
        
        pred_losses.update(pred_loss.cpu().data.numpy())
        
        opt_clf.zero_grad()
        pred_loss.backward()
        opt_clf.step()

    clf.eval()
    for img, label in val_loader:
        imgv = Variable(img).cuda()
        labelv = Variable(label).cuda()
        pred = torch.max(clf(imgv), dim=1)[1]
        val_acc.update((pred==labelv).cpu().data.numpy().sum()/imgv.size(0), n=imgv.size(0))
        
    for img, label in test_loader:
        imgv = Variable(img).cuda()
        labelv = Variable(label).cuda()
        pred = torch.max(clf(imgv), dim=1)[1]
        test_acc.update((pred==labelv).cpu().data.numpy().sum()/imgv.size(0), n=imgv.size(0))
        
    if val_acc.avg > max_acc:
        max_acc = val_acc.avg
        save_models(model_dict, './weights')
        
    print (epoch, pred_losses.avg, val_acc.avg, test_acc.avg)
    sys.stdout.flush()