import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import torch
import torch.nn as nn

from torch.utils import data
from dataloader import DataLoader
from Unet import Unet

#generator
def inf_train_gen():
    while True:
        for img,mask in trainloader:
            yield img,mask
            
gen = inf_train_gen()

hodnoty_treninku=[]
def training ():
  running_loss =0
  for k in range (50):
      
      data,lbl=next(gen)

      data=data.cuda()
      lbl=lbl.cuda()
    
      data.requires_grad=True
      lbl.requires_grad=True
    
      optimizer.zero_grad()   # zero the gradient buffers
    
      net.train()
      #datanp=data.cpu().detach().numpy()
    
      output=net(data)
      #outputnp=data.cpu().detach().numpy() #prevedeni pro zobrazeni
     
      #MSE=torch.nn.MSELoss(size_avarage=False)
      loss=torch.mean((lbl-output)**2)
      loss.backward()  # pocitani gradientu
      optimizer.step() # update parametrs
    
      print(f'{k}/{len(trainloader)}/{epoch}')        #heartbeat, ukaze kolik
      running_loss+=loss.item() #*data.size(0)          #runing loss- soucet lossu v epoche
  running_loss=running_loss/(50)
  hodnoty_treninku.append(running_loss)    
  
  plt.plot(hodnoty_treninku)
  plt.show()
  plt.imshow(output[0,0,:,:].detach().cpu().numpy())
  plt.show()
  plt.imshow(lbl[0,0,:,:].detach().cpu().numpy())


hodnoty_test=[]
def evaluating ():
    with torch.no_grad():       #s vypnutým pocitani gradientu
        running_loss =0
        for kk,(data,lbl) in enumerate(testloader):        
            
             data=data.cuda()
             lbl=lbl.cuda()
            
             net.eval()
    
             output=net(data)
             #outputnp=data.cpu().detach().numpy() 
             
             loss=torch.mean((lbl-output)**2)
             running_loss+=loss.item() #*data.size(0)       #runing loss- soucet lossu v epoche
        running_loss=running_loss/len(testloader)
        hodnoty_test.append(running_loss)    
  
    plt.plot(hodnoty_test)
    plt.show()
    plt.imshow(output[0,0,:,:].detach().cpu().numpy())
    plt.show()
    plt.imshow(lbl[0,0,:,:].detach().cpu().numpy())
    
##############################################################################                    
batch=16  

loader = DataLoader(split='trenink') 
trainloader= torch.utils.data.DataLoader(loader,batch_size=batch, num_workers=8, shuffle=True,drop_last=True)
  
loader = DataLoader(split='test')
testloader= torch.utils.data.DataLoader(loader,batch_size=1, num_workers=2, shuffle=True,drop_last=True)

net=Unet().cuda()
net.requires_grad=True

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

device="cuda" if torch.cuda.is_available() else "cpu"
net=net.to(device)

for epoch in range(10):
  training()
 

  if epoch%1==0:  
     evaluating()
     
     torch.save(net.state_dict(), '/home/ubmi/Documents/data_vse/model.pt')



"""        
#zapis hodnot do grafu     
plt.plot(hodnoty_treninku, 'b')
plt.plot(hodnoty_test,'r')
plt.show

#vykreslit obrazky, puvodni a segmentovany
plt.subplot(121)
plt.imshow(output[0,0,:,:].detach().cpu().numpy())    
plt.subplot(122)        
plt.imshow(lbl[0,0,:,:].detach().cpu().numpy())   

"""