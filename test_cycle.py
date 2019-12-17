import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init

import torch
from Unet import Unet

loader = DataLoader(split='final_test')
finaltestloader= torch.utils.data.DataLoader(loader,batch_size=1, num_workers=2, shuffle=True,drop_last=True)

net = Unet()
net.load_state_dict(torch.load('/home/ubmi/Documents/data_vse/model.pt'), strict=False)


hodnoty_test=[]
def evaluating ():
    with torch.no_grad():       #s vypnutým pocitani gradientu
        running_loss =0
        for kk,(data,lbl) in enumerate(finaltestloader):        
            
             data=data.cuda()
             lbl=lbl.cuda()
            
             net.eval()
    
             output=net(data)
             #outputnp=data.cpu().detach().numpy() 
             
             loss=torch.mean((lbl-output)**2)
             running_loss+=loss.item() #*data.size(0)       #runing loss- soucet lossu v epoche
        running_loss=running_loss/len(finaltestloader)
        hodnoty_test.append(running_loss)    

"""
def dice_loss(pred, target):
  
    smooth = 1.

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
"""