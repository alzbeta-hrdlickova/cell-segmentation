import numpy as np
import matplotlib.pyplot as plt

import torch
import glob
from skimage.io import imread
from torch.utils import data
from scipy import ndimage

class DataLoader(data.Dataset):
    """Nacitani dat"""

    def __init__(self,split="trenink",path_to_data='C:/Users/hrdli/Desktop/DP/data_vse'):
                                      
        self.split=split
        
        #rozdělení do složek
        if self.split == 'trenink':
            self.slozky=glob.glob(path_to_data + '/stage1_train/*')
        elif self.split == 'val':
            self.slozky=glob.glob(path_to_data + '/val/*')
        elif self.split == 'test':
            self.slozky=glob.glob(path_to_data + '/test/*')
                 
        self.maska_list=[]
        self.orig_list=[] 
        
        for i in self.slozky:
            nazev_maska = glob.glob(i + '/masks/*.png')     #názvy obrázků masek
            nazev_orig = glob.glob(i + '/images/*.png')     #názvy původních obrázků
            
            self.orig_list.append(nazev_orig)         
            self.maska_list.append(nazev_maska)
 
    def __len__(self):
        return len(self.orig_list)     #vrací délku datasetu

    def __getitem__(self, idx): 
            
        nazev_maska = self.maska_list[idx]        #seznam cest obrazku
        orig = imread(self.orig_list[idx][0])
        maska = np.zeros((orig.shape[0],orig.shape[1])) 
        #print(self.orig_list[idx])
        
        self.patch_size=[224,224]
        
        for k in nazev_maska:           
            maska_k = imread(k)             #nacteni kazde k-te masky
            dist_map = ndimage.distance_transform_edt(maska_k)
            maska = maska + dist_map        #pricitani distancni mapy
            
        #zmena velikosti obrazku 
        if self.split=='trenink' or self.split=='val':
            out_size=self.patch_size
            in_size=orig.shape
            r1=torch.randint(in_size[0]-out_size[0],(1,1)).view(-1).numpy()[0]
            r2=torch.randint(in_size[1]-out_size[1],(1,1)).view(-1).numpy()[0]
            r=[r1,r2]
        else:
            out_size=orig.shape
            r=[0,0]
        
        #vyindexovani obrazku
        orig=orig[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1],:]
        maska=maska[r[0]:r[0]+out_size[0],r[1]:r[1]+out_size[1]]    
         
        #predelani na float, [1,m,n] pro orig i masku
        image = torch.Tensor(maska.astype(np.float32).reshape((1,out_size[0],out_size[1])))
        image_orig=np.transpose(orig[:,:,0:1].astype(np.float32),(2,0,1))/255-0.5
        image_orig = torch.Tensor(image_orig)
    
        return image,image_orig

if __name__ == "__main__":              #volani fce a vykresleni
    loader = DataLoader(split='trenink')
    trainloader = data.DataLoader(loader,batch_size=2, num_workers=0, shuffle=True,drop_last=True) 
    for it,(mask,orig) in enumerate(trainloader):
        tmp=np.transpose(orig.numpy()[0,:,:,:],[1,2,0])
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(tmp+0.5,cmap="gray")
        fig.add_subplot(1, 2, 2)
        plt.imshow(mask[0,0,:,:],cmap="gray")
        
        break
        
