import numpy as np
import matplotlib.pyplot as plt
from skimage import util 
from skimage.segmentation import watershed
from skimage.morphology import extrema
from skimage.measure import label
from skimage import img_as_float
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
from SEG import SEEGacc
from skimage.segmentation import random_walker
from skimage import measure

''' segmentace predikovaných obrazů pomocí metody rozvodí kontrolované markery a metody náhodého chodce
    vyhodnocení segmentace pomocí funkce SEG '''

SEGS_w=[]
SEGS_rw=[]
SEGS_binar=[]

it=-1
for fin in range(70):
    it+=1
    print(it)

    data =np.load('/Users/betyadamkova/Desktop/final/test - model5/data/' + 'data' + str(it) +'.npy') 
    data=data[0,0,:,:]                                     
    binar_data=data>0
    #binar_data=img_as_float(binar_data)
    label_data = label(binar_data)                                      #označení buněk v masce
    #plt.imshow(label_data,cmap=plt.cm.nipy_spectral)
     
    output =np.load('/Users/betyadamkova/Desktop/final/test - model5/output/' + 'output' + str(it) +'.npy') 
    output=output[0,0,:,:]
    inverse_output=util.invert(output)                                  #inverzní distanšní mapa 
    binar_output=output > -0.26                       
    binar_output=remove_small_holes(remove_small_objects(binar_output, 3),1200) 
    #plt.imshow(inverse_output,cmap="gray")
    
    ################### marker controlled watershed
    h =0.02                                                             #h= minimální výška extrahovaných maxim
    h_maxima = extrema.h_maxima(output, h)                              #nalezení středu buněk
    marker  = label(h_maxima)                                           #označení stredu buněk
    #plt.imshow(h_maxima,cmap="gray")                                     
    labels=watershed(inverse_output, marker, mask=binar_output)         #vstup do funkce: inverní distanční mapa, označené středy, binární
    #plt.imshow(labels,cmap=plt.cm.nipy_spectral)
    
    ################ random walker segmentation
    h =0.02
    h_maxima = extrema.h_maxima(output, h)            #středy buněk 
    marker2  = label(h_maxima)                         #označení buněk
    marker2[~binar_output] = -1                         #bunky=0, okoli=-1, středy buněk označené
    #plt.imshow(marker2,cmap="gray")                        
    if (np.amax(marker2)).astype(np.int)==-1:           #pokud na obrazu není žádná buňka, random walker nelze volat
        labels_rw=binar_output
    else:
        labels_rw = random_walker(binar_output, marker2, beta=24, mode='bf')
    #plt.imshow(labels_rw,cmap=plt.cm.nipy_spectral)

    ################ měření velikosti označených buněk
    properties_maska = measure.regionprops(label_data)
    properties_rw = measure.regionprops(labels_rw)
    properties_labels = measure.regionprops(labels)
    #print([prop.area for prop in properties_maska])
    #print([prop.area for prop in properties_labels])
    #print([prop.area for prop in properties_rw])

    ################# vyhodnocení segmentace watershed - marker, random walker a binární obrazy
    seg_score=SEEGacc(labels, label_data)
    SEGS_w.append(seg_score)
    print("SEG_w:", seg_score) 

    seg_score2=SEEGacc(labels_rw, label_data)
    SEGS_rw.append(seg_score2)
    print("SEG_rw:", seg_score2)

    seg_score3=SEEGacc(binar_output, binar_data)
    SEGS_binar.append(seg_score3)
    print("SEG_binar:", seg_score3)
   
segs_mean1=sum(SEGS_w)/70
segs_mean2=sum(SEGS_rw)/70
segs_mean3=sum(SEGS_binar)/70
print("Výsledek SEG watershed:", segs_mean1)  
print("Výsledek SEG random walker:", segs_mean2) 
print("Výsledek SEG binarních obrazů:", segs_mean3) 

#vykreslení
fig, axes = plt.subplots(ncols=3, figsize=(15, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(label_data, cmap=plt.cm.nipy_spectral)
ax[0].set_title('a)')
ax[0].axis('off')
ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('b)')
ax[1].axis('off')
ax[2].imshow(labels_rw, cmap=plt.cm.nipy_spectral)
ax[2].set_title('c)')
ax[2].axis('off')
