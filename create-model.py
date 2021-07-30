import numpy as np
import matplotlib.pyplot as plt
import torch
from dataloader import DataLoader
from Unet import Unet
from torch.utils import data
from torch.utils.data import random_split
import torch.nn as nn

"""model neuronové sítě, učení, validace a testování modelu, volání funkce DataLoader
čerpání informací z 11. cvičení MPC-MLR """

batch=16

dataset = DataLoader(split="trenink")
trainset, valset, test= random_split(dataset, [530,70,70])   #rozdělení dat na trénovací, validační a testovací 

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(
    valset, batch_size=batch, num_workers=1, shuffle=True, drop_last=True)
finaltestloader = torch.utils.data.DataLoader(
    test, batch_size=1, num_workers=1, shuffle=True, drop_last=True)  

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

if __name__ == "__main__":   
    it=-1
    for epoch in range(40):
      for k,(data,lbl) in enumerate(trainloader):       
        it+=1
        print(it)
        
        data=data.cpu()
        lbl=lbl.cpu()
        
        data.requires_grad=True
        lbl.requires_grad=True
        
        optimizer.zero_grad()   # nulovani gradientu
        net.train()
        output=net(lbl)
        
        loss=torch.mean((data-output)**2)     #nn.MSELoss(output,data)
        loss.backward()     #vypocet gradientu
        optimizer.step()    #aktualizace parametru
        
        train_loss_tmp.append(loss.detach().cpu().numpy())
        
        if it%20==0:
          for kk,(data,lbl) in enumerate(testloader):
              
              data=data.cpu()
              lbl=lbl.cpu()
    
              net.eval()
              output=net(lbl)
    
              loss=torch.mean((data-output)**2)
              test_loss_tmp.append(loss.detach().cpu().numpy())
                       
      train_loss.append(np.mean(train_loss_tmp))
      test_loss.append(np.mean(test_loss_tmp))
      position.append(epoch)
    
      train_loss_tmp=[]
      test_loss_tmp=[]
          
      fig = plt.figure()
      plt.plot(position,train_loss, label="chyba při učení")   #oznaceni legendy
      plt.plot(position,test_loss,label="validační chyba")
      plt.legend()
      plt.ylabel('Chybová funkce')  #označení os
      plt.xlabel('Epocha')
      plt.show()
          
      plt.savefig('images/training_loss.png')
      plt.close("all")

torch.save(net.state_dict(), 'C:/Users/HP/Desktop/Bety/python/images/model10.pt')

print('Training finished!!!')

################################################################# testování
device = torch.device("cpu")
net = Unet()                                           #instancování třídy vytvořené v modelu
net.load_state_dict(torch.load('C:/Users/HP/Desktop/Bety/python/images/model9.pt',map_location=torch.device('cpu')))  #volání natrénovaného modelu
net=net.to(device)
#my_model = net.load_state_dict(torch.load('C:/Users/HP/Desktop/Bety/python/images/model.pt', map_location=torch.device('cpu')))

if __name__ == "__main__":
    it=-1
    for jj,(data,lbl) in enumerate(finaltestloader):
        it+=1
        print(it)
              
        data=data.cpu()
        lbl=lbl.cpu()
        net.eval()
        output=net(lbl)     
        
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(data[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 3, 2)
        plt.imshow(output[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 3, 3)
        plt.imshow(lbl[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
      
        #prevest na np array,ulozit
        data = data.data.cpu().numpy()
        np.save('images/data/data'+ str(it), data)
        output = output.data.cpu().numpy()
        np.save('images/output/output' + str(it), output)
        lbl = lbl.data.cpu().numpy()
        np.save('images/lbl/lbl'+ str(it), lbl)
        
