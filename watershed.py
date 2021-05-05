import numpy as np
import matplotlib.pyplot as plt
from skimage import util 
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.morphology import extrema
from skimage.measure import label
from skimage import data, util, filters, color
from skimage.feature import peak_local_max
from skimage import img_as_bool
from skimage import img_as_float
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
from SEG import SEEGacc
from skimage.segmentation import random_walker
from skimage import measure
from skimage.morphology import local_maxima

SEGS=[]
SEGS_rw=[]
SEGS_binar=[]

it=-1
for fin in range(70):
    it+=1
    print(it)

    data =np.load('/Users/betyadamkova/Desktop/final/data/' + 'data' + str(it) +'.npy') 
    data=data[0,0,:,:]
    #inverse_data=util.invert(data)             #inverzní distanšční mapa
    #h1 = 1                                      #hledání středů buněk, h= minimální výška extrahovaných maxim
    #h1_maxima = extrema.h_maxima(data, h1)
    #plt.imshow(h1_maxima,cmap="gray")
    binar_data=data>0
    #binar_data=remove_small_holes(remove_small_objects(binar_data))
    binar_data=img_as_float(binar_data)
    label_h_maxima1 = label(binar_data)        #označení buněk
    plt.imshow(label_h_maxima1,cmap=plt.cm.nipy_spectral)
    #data_edge=data==1               #okraje buněk
    #data_edge=remove_small_holes(remove_small_objects(data_edge, 1), 1)
    #plt.imshow(data_edge,cmap="gray")
    
    output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy') 
    output=output[0,0,:,:]
    binar_output=output > -0.32                            
    binary_output=remove_small_holes(remove_small_objects(binar_output, 50),50)

    h2 =0.05    
    h2_maxima = extrema.h_maxima(output, h2)        #malezení středu buněk
    #seed=local_maxima(distance_output, selem=None, connectivity=50, indices=False, allow_borders=False)
    #peak=peak_local_max(distance_output,min_distance=7, indices=False,footprint=np.ones((9,9)), labels=binar_output)
    label_h_maxima2  = label(h2_maxima) 
    plt.imshow(h2_maxima,cmap="gray")                                         #označení buněk
    
    labels=watershed(-output, label_h_maxima2, mask=binar_output) #inverní distanční mapa, označené středy, binární
    plt.imshow(labels,cmap=plt.cm.nipy_spectral)
    
    
    ################random walker for segmentation
    
    h2 =0.05    
    h2_maxima = extrema.h_maxima(output, h2)
    marker2  = label(h2_maxima)
    marker2[~binar_output] = -1
    labels_rw = random_walker(binar_output, marker2, beta=10, mode='bf')
    plt.imshow(labels_rw,cmap=plt.cm.nipy_spectral)
    
    ################ měření velikosti označených buněk
    properties_maska = measure.regionprops(label_h_maxima1)
    properties_rw = measure.regionprops(labels_rw)
    properties_labels = measure.regionprops(labels)
    print([prop.area for prop in properties_maska])
    print([prop.area for prop in properties_labels])
    print([prop.area for prop in properties_rw])
 
    
############################### vyhodnocení segmentace pomocí watershed - marker, random walker, a binární obrazy
    seg_score=SEEGacc(labels, label_h_maxima1)
    SEGS.append(seg_score)
    #print("SEG:", seg_score) 

    seg_score2=SEEGacc(labels_rw, label_h_maxima1)
    SEGS_rw.append(seg_score2)
    #print("SEG_binar:", seg_score3)

    seg_score3=SEEGacc(binar_output, binar_data)
    SEGS_binar.append(seg_score3)
    #print("SEG_binar:", seg_score3)
    
segs_mean=sum(SEGS)/70
segs_mean2=sum(SEGS_rw)/70
segs_mean3=sum(SEGS_binar)/70
print("Výsledek SEG:", segs_mean)  
print("Výsledek SEG_rw:", segs_mean2) 
print("Výsledek SEG_binar:", segs_mean3) 

#vykreslení
fig, axes = plt.subplots(ncols=3, figsize=(15, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(label_h_maxima1, cmap=plt.cm.nipy_spectral)
ax[0].set_title('a)')
ax[0].axis('off')
ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('b)')
ax[1].axis('off')
ax[2].imshow(labels_rw, cmap=plt.cm.nipy_spectral)
ax[2].set_title('c)')
ax[2].axis('off')

