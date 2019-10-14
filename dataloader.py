import numpy as np
import matplotlib.pyplot as plt

import torch
import glob
from skimage.io import imread
from torch.utils import data

class Dataloader(Dataset):
    """Nacitani dat"""

    def __init__(self,split="trenink",path_to_data='/Desktop/data'):
        
        self.split=split
        self.path=path_to_data + '/' + split 
        
        #self.pic1 = mpimg.imread('pic1.png')
        #self.mask1 = mpimg.imread('mask1.png')
     
        self.file_list=[]
        self.lbls=[]

     for k in range(6):
        #prochazeni slozky
        files=glob.glob(self.path + '/'+ str(k) +'/*png')
        self.file_list.extend(files)
        self.lbls.extend([k]*len(files))
        
        self.num_of_imgs=len(self.file_list)
        
    
    def __len__(self):
        return len(self.num_of_imgs)



    def __getitem__(self, idx):
        
        img=imread(self.file_list[idx])
        img=torch.Tensor(img.astype(np.float32)/255-0.5)
        lbl=torch.Tensor(np.array(self.lbls[idx]).astype(np.float32))
        return img,lbl

loader = Dataloader(split='trenink')
trainloader= data.Dataloader(loader,batch_size=2, num_workers=0, shuffle=True,drop_last=True)







for it,(batch,lbls) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
  print(batch)
  print(batch.size())
  print(lbls)
  plt.imshow(batch[0,:,:].detach().cpu().numpy())
  break
        
        
        
        
  """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
 """