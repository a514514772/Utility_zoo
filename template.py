import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image

%matplotlib inline
import matplotlib.pyplot as plt

REPLACE_THIS_WITH_ITS_SIZE = 3
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# define your loader here
# loader = ...

class MotionNet(nn.Module):
    def __init__(self, input_dim):
        super(MotionNet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, input_dim//2),
                                nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(input_dim//2, 3))
    
    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        
        # delta x, y, z
        return output

feature_extractor = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2]).cuda()
motion_net = MotionNet(REPLACE_THIS_WITH_ITS_SIZE).cuda()
print (feature_extractor)
#print (resnet18)

opt_fe = optim.SGD(feature_extractor.parameters(), lr=1e-3, momentum=0.9)
opt_mn = optim.SGD(motion_net.parameters(), lr=1e-3, momentum=0.9)

for it in range(max_epochs):
    for index, (img1, img2, label) in enumerate(data_loader):
        img1v = Variable(img1).cuda()
        img2v = Variable(img2).cuda()
        labelv = Variable(label).cuda()
        
        feat1 = feature_extractor(img1v).view(feat1.size(0), -1)
        feat2 = feature_extractor(img2v).view(feat2.size(0), -1)
        
        pred = motion_net(torch.cat([feat1, feat2], 1))
        
        # define your loss here
        loss = YOUR_LOSS_FUNCTION(pred)
        
        opt_fe.zero_grad()
        opt_mn.zero_grad()
        
        loss.backward()
        
        opt_fe.step()
        opt_mn.step()