import numpy
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch
import glob
from skimage.io import imread
from torch.utils import data

class Dataloader(data.Dataset):
    """Nacitani dat"""

    def __init__(self,path_to_data='/Users/betyadamkova/Desktop/data_vse/stage1_train'):
        
        self.path=path_to_data + '/' 
     
        self.maska_list=[]
        self.orig_list=[]
        
        self.slozka= glob.glob(path_to_data + '/*') 
        
        for i in self.slozka:
            nazev_maska = glob.glob(i + '/masks/*.png')
            nazev_orig = glob.glob(i + '/images/*.png')
            print(nazev_orig)
            orig = imread(nazev_orig[0])
            
            self.orig_list.append(orig)
        
            
            maska = numpy.zeros((256,256))
            
            for k in nazev_maska:
                 #prochazeni slozky
                maska_k = imread(k)
                maska = maska + maska_k
                
                
            self.maska_list.append(maska)
           
            """   
            imgplot = plt.imshow(maska)
            plt.show()
            img.imsave('maska.png',maska,cmap="gray")     
            """ 
                
    
    def __len__(self):
        return len(self.maska_list)



    def __getitem__(self, idx):     
        
        #img_mask = torch.Tensor(img_mask.astype(np.float32)/255-0.5)
        #lbl=torch.Tensor(np.array(self.lbls[idx]).astype(np.float32))
      
        return self.maska_list[idx],self.orig_list[idx]

loader = Dataloader()
#trainloader= data.Dataloader(loader,batch_size=2, num_workers=0, shuffle=True,drop_last=True)


"""

for it,(batch,lbls) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
  print(batch)
  print(batch.size())
  print(lbls)
  plt.imshow(batch[0,:,:].detach().cpu().numpy())
  break
        
"""     
