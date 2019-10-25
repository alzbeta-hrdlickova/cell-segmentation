import numpy
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch
import glob
from skimage.io import imread
from torch.utils import data
from scipy.spatial import distance

class Dataloader(data.Dataset):
    """Nacitani dat"""

    def __init__(self,path_to_data='/Users/betyadamkova/Desktop/data_vse/stage1_train'):
     
        self.maska_list=[]
        self.orig_list=[]
        
        self.slozka= glob.glob(path_to_data + '/*') 
        
        for i in self.slozka:
            nazev_maska = glob.glob(i + '/masks/*.png')     #názvy obrázků masek
            nazev_orig = glob.glob(i + '/images/*.png')     #názvy původních obrázků
            #print(nazev_orig)
            
            orig = imread(nazev_orig[0])
            self.orig_list.append(orig)         #přidání orig do orig_list
            
            dim1 = orig.shape[0]
            dim2 = orig.shape[1]
            maska = numpy.zeros((dim1,dim2))    #vytvoření matice nul o velikosti orig

    #ulozit nazvy masek, bez imread = nazev_maska   
                
    
    def __len__(self):
        return len(self.orig_list)     #vrací délku datasetu


    def __getitem__(self, idx): 
            
    #seznam cest obrazku
        for k in nazev_maska:           #procházení masek a sčítání dohromady
            maska_k = imread(k)
            maska = maska + maska_k
                
        self.maska_list.append(maska)
        
     #distancni mapu, regresovat 
        dist_map_list = []
        dist_map = distance.euclidean(maska_list)
        dist_map_list.append(dist_map)
                       
     #predelat na float, batch [1,m,n] pro orig i masku
        image = imread(self.maska_list[idx])
        image = torch.Tensor(image.astype(numpy.float32)/255-0.5)
        image_orig = torch.Tensor(numpy.array(self.orig_list[idx]).astype(numpy.float32))
        return image,image_orig
      
loader = Dataloader()
#trainloader = data.Dataloader(loader,batch_size=2, num_workers=0, shuffle=True,drop_last=True)


"""

for it,(batch,lbls) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
  print(batch)
  print(batch.size())
  print(lbls)
  plt.imshow(batch[0,:,:].detach().cpu().numpy())
  break
        
"""     
