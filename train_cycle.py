import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import torch
import torch.nn as nn

from torch.utils import data
from dataloader import DataLoader
from Unet import Unet


batch=1  #batch alespon 16

loader = DataLoader(split='trenink') 
trainloader= torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=0, shuffle=True,drop_last=True)
  
loader = DataLoader(split='test')
testloader= torch.utils.data.DataLoader(loader,batch_size=1, num_workers=0, shuffle=True,drop_last=True)

#device="cuda" if torch.cuda.is_available() else "cpu"
#model=model.to(device)

net=Unet()      #.cuda()
net.requires_grad=True

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

#vytvorit list pro vysledky loss, každou 50-tou hodnotu, vykreslit do grafu
hodnoty_train=[]
hodnoty_test=[]

it=-1
for epoch in range(1000):
  for k,(data,lbl) in enumerate(trainloader):
    it+=1
    
    data=data       #.cuda()
    lbl=lbl         #.cuda()
    
    data.requires_grad=True
    lbl.requires_grad=True
    
    optimizer.zero_grad()   # zero the gradient buffers
    
    net.train()
    
    output=net(data)

    datanp=data.cpu().detach().numpy()
    
    output=net(data)
    outputnp=data.cpu().detach().numpy() #prevedeni pro zobrazeni
     
    #MSE=torch.nn.MSELoss(size_avarage=False)
    loss=torch.mean((lbl-output)**2)
    loss.backward()  # pocitani gradientu
    optimizer.step() # update parametrs
    
    

    #jednou za určitý počet epoch je potřeba změřit výsledky
    #to stejné jako při trénování ale bez výpočtu gradientu úpravy vah
    if it%50==0:  
      for kk,(data,lbl) in enumerate(testloader):
          
          net=Unet()          #.cuda()
          net.requires_grad=True
            
          optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            
          it=-1
          for epoch in range(1000):
             for k,(data,lbl) in enumerate(trainloader):
                 it+=1
                
                 data=data       #.cuda()
                 lbl=lbl         #.cuda()
                
                 data.requires_grad=True
                 lbl.requires_grad=True
                
                 optimizer.zero_grad()   # zero the gradient buffers
                    
                 net.eval()
               
                 output=net(data)
            
                 datanp=data.cpu().detach().numpy()
                
                 output=net(data)
                 outputnp=data.cpu().detach().numpy() 
                 
                 loss=torch.mean((lbl-output)**2)
                 ##loss.backward() 
                 optimizer.step() 
"""        
#zapis hodnot do grafu     
hodnoty_train.append(output[0,0,:,:].data.cpu().numpy())`
hodnoty_test.append(lbl[0][0].data.cpu().numpy())`

#vykreslit obrazky, puvodni a segmentovany
plt.subplot(121)
plt.imshow(outputnp)    #output[1][0].cpu().numpy()
plt.subplot(122)
plt.imshow(lbl)         #(data[1].data.cpu().permute(1,2,0).numpy()*0.5+0.5))

"""