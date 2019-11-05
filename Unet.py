import numpy
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
import torchvision
from torchvision import transforms

import glob
import os
from skimage.io import imread

from torch.utils import data
from dataloader import DataLoader

batch=16

loader = DataLoader(split='trenink') 
trainloader= torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=True,drop_last=True)

  
loader = DataLoader(split='test')
testloader= torch.utils.data.DataLoader(loader,batch_size=1, num_workers=0, shuffle=True,drop_last=True)


net=Unet().cuda()

optimizer = optim.Adam(net.parameters(), lr=0.001)


it=-1
for epoch in range(1000):
  for k,(data,lbl) in enumerate(trainloader):
    it+=1
    
    data=data.cuda()
    lbl=lbl.cuda()
    
    
    data.requires_grad=True
    lbl.requires_grad=True
    
    optimizer.zero_grad()   # zero the gradient buffers
    
    net.train()
    
    output=net(data)
    output=F.sigmoid(output)
    
    loss=dice_loss(output, lbl) ### tady budete mít MSE asi místo toho
    
    loss.backward()  ## claculate gradients
    optimizer.step() ## update parametrs
    
    
"""      
    if it%50==0:  ###jednou za určitý počet epoch je potřeba změřit výsledky
      for kk,(data,lbl) in enumerate(testloader):
 
#to stejné jako při trénování ale bez výpočtu gradientu úpravy vah
"""