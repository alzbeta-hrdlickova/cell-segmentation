import numpy as np
import matplotlib.pyplot as plt
import torch
from Unet import Unet
from torch.utils import data
from dataloader import DataLoader
from torch.utils.data import random_split

dataset = DataLoader(split="trenink")
trainset, valset, test= random_split(dataset, [530,70,70])   #rozdělení dat na trénovací, validační a testovací 

finaltestloader = torch.utils.data.DataLoader(
    test, batch_size=1, num_workers=1, shuffle=True, drop_last=True) 

device = torch.device("cpu")
net = Unet()                                                               #instancování třídy vytvořené v modelu
net.load_state_dict(torch.load('C:/Users/HP/Desktop/Bety/python/images/model.pt',map_location=torch.device('cpu')))  #volání natrénovaného modelu
net=net.to(device)
#my_model = net.load_state_dict(torch.load('C:/Users/HP/Desktop/Bety/python/images/model.pt', map_location=torch.device('cpu')))

if __name__ == "__main__":
    it=-1
    for jj,(lbl,data) in enumerate(finaltestloader):
        it+=1
        print(it)
              
        data=data.cpu()
        lbl=lbl.cpu()
        net.eval()
        output=net(data)     
        
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(data[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 3, 2)
        plt.imshow(lbl[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
        fig.add_subplot(1, 3, 3)
        plt.imshow(output[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
      
        #prevest na np array,ulozit
        data = data.data.cpu().numpy()
        np.save('images/data/data'+ str(it), data)
        output = output.data.cpu().numpy()
        np.save('images/output/output' + str(it), output)
        lbl = lbl.data.cpu().numpy()
        np.save('images/lbl/lbl'+ str(it), lbl)
        
