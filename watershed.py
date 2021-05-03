import numpy as np
import matplotlib.pyplot as plt
from skimage import util 
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.morphology import extrema
from skimage.measure import label
from skimage import data, util, filters, color
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.filters import threshold_yen
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
from SEG import SEEGacc

def binar(data):
    threshold=threshold_otsu(data)
    for x in range(224):      
        for y in range(224):
            if data[x,y] >= threshold:
                data[x,y] = 1
            else:
                data[x,y] = 0
    return data

def edge_binar(labels):
    threshold2=threshold_yen(labels)
    for x in range(224):      
        for y in range(224):
            if labels[x,y] > threshold2:
                labels[x,y] = 1
            else:
                labels[x,y] = 0
    return labels

def edge_data(edges1):
    for x in range(224):      
        for y in range(224):
            if edges1[x,y] > 1:
                edges1[x,y] = 0
            elif edges1[x,y] < 1:
                edges1[x,y] = 0
            else:
                edges1[x,y] = 1
    return edges1

SEGS=[]
it=-1
for fin in range(70):
    it+=1
    print(it)

    data =np.load('/Users/betyadamkova/Desktop/final/data/' + 'data' + str(it) +'.npy') 
    data=data[0,0,:,:]
    inverse_data=util.invert(data)              #inverzní distanšční mapa
    h1 = 0
    h1_maxima = extrema.h_maxima(data, h1)
    label_h_maxima1 = label(h1_maxima)
    data_edge=edge_data(data) 
    plt.imshow(data_edge,cmap="gray")
    
    output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy') 
    output=output[0,0,:,:]
    plt.imshow(output,cmap="gray")
    binar_output=binar(output)
    h2 = 0.005
    h2_maxima = extrema.h_maxima(output, h2)
    label_h_maxima2 = label(h2_maxima)
    
    labels=watershed(inverse_data, label_h_maxima1, mask=binar_output)
    edges = filters.sobel(labels)
    output_edge=edge_binar(edges)
    
    seg_score=SEEGacc (labels, data, 0)
    SEGS.append(seg_score)
    
segs_mean=sum(SEGS)/70
print("SEG:", segs_mean) 

fig, axes = plt.subplots(ncols=4, figsize=(15, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(inverse_data, cmap=plt.cm.gray)
ax[0].set_title('Inverzní distanční mapa')
ax[0].axis('off')
ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('Watershed')
ax[1].axis('off')
ax[2].imshow(data_edge, cmap=plt.cm.gray)
ax[2].set_title('Edges data')
ax[2].axis('off')
ax[3].imshow(output_edge, cmap=plt.cm.gray)
ax[3].set_title('Output edge')
ax[3].axis('off')
