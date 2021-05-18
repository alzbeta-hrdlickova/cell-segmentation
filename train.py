import numpy as np
import matplotlib.pyplot as plt
import torch
from dataloader import DataLoader
from Unet import Unet
from torch.utils import data
from torch.utils.data import random_split
import torch.nn as nn

"""model neuronové sítě, učení a validace modelu, volání funkce DataLoader
čerpání informací z 11. cvičení MPC-MLR https://colab.research.google.com/drive/1-Wa6iWwK39Gxm_LptwalRGA_CwEDKXdG#scrollTo=l2aV6i_gaxrs """

batch=16

dataset = DataLoader(split="trenink")
trainset, valset, test= random_split(dataset, [530,70,70])   #rozdělení dat na trénovací, validační a testovací 

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(
    valset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)

net=Unet().cpu()
net.requires_grad=True

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
device="cuda" if torch.cuda.is_available() else "cpu"
net=net.to(device)

train_loss=[]
test_loss=[]
position=[]
train_loss_tmp=[]
test_loss_tmp=[]

num_epoch=40

if __name__ == "__main__":
    it=-1
    for epoch in range(num_epoch):
      for k,(lbl,data) in enumerate(trainloader):       #lbl=maska data=origimg
        it+=1
        print(it)
        
        data=data.cpu()
        lbl=lbl.cpu()
        
        data.requires_grad=True
        lbl.requires_grad=True
        
        optimizer.zero_grad()  
        net.train()
        output=net(data)
        
        loss=torch.mean((lbl-output)**2)     #nn.MSELoss(output,data)
        loss.backward()     
        optimizer.step()    #aktualizace parametru
        
        train_loss_tmp.append(loss.detach().cpu().numpy())
        
        if it%20==0:
          for kk,(lbl,data) in enumerate(testloader):
              
              data=data.cpu()
              lbl=lbl.cpu()
    
              net.eval()
              output=net(data)
    
              loss=torch.mean((lbl-output)**2)
              test_loss_tmp.append(loss.detach().cpu().numpy())
              
              
      train_loss.append(np.mean(train_loss_tmp))
      test_loss.append(np.mean(test_loss_tmp))
      position.append(epoch)
    
      train_loss_tmp=[]
      test_loss_tmp=[]
          
      fig = plt.figure()
      plt.plot(position,train_loss, label="chyba při učení")  
      plt.plot(position,test_loss,label="validační chyba")
      plt.legend()
      plt.ylabel('Chybová funkce MSE')  #označení os
      plt.xlabel('Epocha')
      plt.show()
          
      plt.savefig('images/training_loss.png')
      plt.close("all")

torch.save(net.state_dict(), 'C:/Users/HP/Desktop/Bety/python/images/model8.pt')

print('Training finished!!!')
