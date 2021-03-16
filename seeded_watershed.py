import numpy as np
import matplotlib.pyplot as plt
from skimage import util 
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.morphology import extrema
from skimage.measure import label
from skimage import data, util, filters, color
from skimage.feature import peak_local_max

#marker controlled watershed - distační mapa, binár a seed -obracená distanční mapa se seedama, maska pro definici pozadí

def binar(data):
    #data=data[0,0,:,:]
    #fig, ax = try_all_threshold(data, figsize=(10, 6), verbose=False)    #najít optimální threshold
    threshold=threshold_otsu(data)
    for x in range(224):      
        for y in range(224):
            if data[x,y] >= threshold:
                data[x,y] = 1
            else:
                data[x,y] = 0
    #return data

def edge_binar(labels):
    
    for x in range(224):      
        for y in range(224):
            if labels[x,y] > 0:
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

it=69
data =np.load('/Users/betyadamkova/Desktop/final/data/' + 'data' + str(it) +'.npy') 
data=data[0,0,:,:]
inverse_data=util.invert(data)              #inverzní distanšční mapa
h1 = 0
h1_maxima = extrema.h_maxima(data, h1)
label_h_maxima1 = label(h1_maxima)
plt.imshow(data,cmap="gray")
binar_edge=edge_data(data)
plt.imshow(binar_edge,cmap="gray")

it=69
output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy') 
output=output[0,0,:,:]
binar_output=binar(output)
#inverse_output=util.invert(output)  #inverzní distanšční mapa
h2 = 0.005
h2_maxima = extrema.h_maxima(output, h2)
label_h_maxima2 = label(h2_maxima)

fig = plt.figure()
fig.add_subplot(1, 2, 1) 
plt.imshow(inverse_data,cmap="gray")
fig.add_subplot(1, 2, 2) 
plt.imshow(label_h_maxima2,cmap=plt.cm.nipy_spectral)

labels=watershed(inverse_data, label_h_maxima1, mask=binar_output)
edges = filters.sobel(labels)
#binar_edge=edge_binar(edges)

fig, axes = plt.subplots(ncols=4, figsize=(17, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(inverse_data, cmap=plt.cm.gray)
ax[0].set_title('Inverzní distanční mapa')
ax[0].axis('off')
ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('Watershed')
ax[1].axis('off')
ax[2].imshow(binar_edge, cmap=plt.cm.gray)
ax[2].set_title('Edges data')
ax[2].axis('off')
ax[3].imshow(edges, cmap=plt.cm.gray)
ax[3].set_title('Output edge')
ax[3].axis('off')



##############################################################
edges = filters.sobel(labels)
plt.imshow(edges,cmap="gray")

grid = util.regular_grid(data.shape, n_points=468)
seeds = np.zeros(data.shape, dtype=int)

seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
w0 = watershed(edges, seeds)
w1 = watershed(edges, seeds, compactness=0.01)

fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.imshow(color.label2rgb(w0, data, bg_label=-1))
ax0.set_title('Classical watershed')

ax1.imshow(color.label2rgb(w1, data, bg_label=-1))
ax1.set_title('Compact watershed')

plt.show()
