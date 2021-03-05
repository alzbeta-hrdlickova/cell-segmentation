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
           

it=-1
for fin in range(70):
    it=+1
    
    lbl =np.load(path_to_file + 'lbl' + str(it) +'.npy') 
    #lbl + int(it)=lbl
    for x in range(lbl[0,0,:,:].shape[0]):
        for y in range(lbl[0,0,:,:].shape[1]):
            if lbl[0,0,:,:][x,y] < -0.45 :
                lbl[0,0,:,:][x,y] == 0
            else:
                lbl[0,0,:,:][x,y] == 1
                
    #np.amin(lbl, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

    output  =np.load(path_to_file + 'output' + str(it) +'.npy')
    for x in range(output[0,0,:,:].shape[0]):
        for y in range(output[0,0,:,:].shape[1]):
            if output[0,0,:,:][x,y] < -0.35 :
                output[0,0,:,:][x,y] == 0
            else:
                output[0,0,:,:][x,y] == 1
                
    #np.amin(output, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

TP = np.sum(((lbl==1) & (output ==1)).astype(np.float32))
FN = np.sum(((lbl==0) & (output ==0)).astype(np.float32))
TN = 1 - FN
FP = 1 - TP

sensitivity = TP / (TP + FN)
specificity = TN / (FP + TN)
#precision =  TP/ (FP + TP)
#accuracy = (TN + TP)/ (TN + FP + FN + TP)
print(sensitivity)
print(specificity)    
    
