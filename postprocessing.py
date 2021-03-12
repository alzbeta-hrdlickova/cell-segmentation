import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def binar(data):
    data=data[0,0,:,:]
    #fig, ax = try_all_threshold(data, figsize=(10, 6), verbose=False)    #najít optimální threshold
    threshold=threshold_otsu(data)
    for x in range(224):      
        for y in range(224):
            if data[x,y] >= threshold:
                data[x,y] = 1
            else:
                data[x,y] = 0
    return data

it=-1
for fin in range(70):
    it+=1
    print(it)

    data =np.load('/Users/betyadamkova/Desktop/final/data/' + 'data' + str(it) +'.npy') 
    binar_data=binar(data)
    
    output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy')
    binar_output=binar(output)
    
    image = np.logical_or(binar_output,binar_output)

    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    
fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
