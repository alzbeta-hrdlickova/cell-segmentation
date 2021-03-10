import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

import glob
import os
from skimage.io import imread
from dataloader import DataLoader
from Unet import Unet
from torch.utils import data
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix
from scipy.signal import argrelextrema

path_to_file='images/final/'                #nacitani dat
#final= glob.glob(path_to_file + '*')

fig = plt.figure()
fig.add_subplot(1, 2, 1)   
lbl0=np.load('images/final/lbl0.npy')
plt.imshow(lbl0[0,0,:,:],cmap="gray")
fig.add_subplot(1, 2, 2)
output0=np.load('images/final/output0.npy')
plt.imshow(output0[0,0,:,:],cmap="gray")

lbl_set=[]   
output_set=[]   
dice_set=[] 
sensitivity_set=[] 
specificity_set=[] 
FP_set=[] 
FN_set=[] 

 
it=-1
for fin in range(70):
    it+=1
    #print(it)
    lbl =np.load(path_to_file + 'lbl' + str(it) +'.npy') 
    lbl_set.append(lbl[0,0,:,:])
    """
    for x in range(lbl[0,0,:,:].shape[0]):
        for y in range(lbl[0,0,:,:].shape[1]):
            
            if lbl[0,0,:,:][x,y] < -0.45 :
                lbl[0,0,:,:][x,y] = 1
            else:
                lbl[0,0,:,:][x,y] = 0     
    """
    output =np.load(path_to_file + 'output' + str(it) +'.npy')
    output_set.append(output[0,0,:,:])
    dice_dice=dice_loss(output,lbl)
    dice_set.append(dice_dice)
    final_dice=sum(dice_set)/70
    """
    for x in range(output[0,0,:,:].shape[0]):
        for y in range(output[0,0,:,:].shape[1]):
            if output[0,0,:,:][x,y] < -0.35 :
                output[0,0,:,:][x,y] = 1
            else:
                output[0,0,:,:][x,y] = 0
    """
    TP = np.sum(((lbl >= (-0.45)) & (output >= (-0.35))).astype(np.float32))
    FN = np.sum(((lbl < (-0.45)) & (output < (-0.35))).astype(np.float32))
    TN = 1 - FN
    FP = 1 - TP  
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    sensitivity_set.append(sensitivity)
    specificity_set.append(specificity)
    final_sensitivity=sum(sensitivity_set)/70
    final_specificity=sum(specificity_set)/70
    
print(final_sensitivity)
print(final_specificity)
print(final_dice)
      
#precision =  TP/ (FP + TP)
#accuracy = (TN + TP)/ (TN + FP + FN + TP)
#Dice = 2*TP / ( 2*TP + FP + FN )
#Jaccard = TP / ( TP + FP + FN )   
