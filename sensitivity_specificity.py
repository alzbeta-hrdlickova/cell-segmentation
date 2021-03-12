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

sensitivity_set=[] 
specificity_set=[]
accuracy_set=[]
dice_set=[]
jaccard_set=[]

it=-1
for fin in range(70):
    it+=1
    print(it)

    data =np.load('/Users/betyadamkova/Desktop/final/data/' + 'data' + str(it) +'.npy') 
    binar_data=binar(data)
    
    output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy')
    binar_output=binar(output)
    
    TP = np.sum(((data==1) & (output ==1)).astype(np.float32))
    FN = np.sum(((data==1) & (output ==0)).astype(np.float32))
    TN = np.sum(((data==0) & (output ==0)).astype(np.float32))
    FP = np.sum(((data==0) & (output ==1)).astype(np.float32))
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TN + TP)/ (TN + FP + FN + TP)
    Dice = 2*TP / ( 2*TP + FP + FN )
    Jaccard = TP / ( TP + FP + FN ) 
    #precision =  TP/ (FP + TP)
    
    sensitivity_set.append(sensitivity)
    specificity_set.append(specificity)
    accuracy_set.append(accuracy)
    dice_set.append(Dice)
    jaccard_set.append(Jaccard)
    
    final_sensitivity=sum(sensitivity_set)/70
    final_specificity=sum(specificity_set)/70
    final_accuracy=sum(accuracy_set)/70
    final_dice=sum(dice_set)/70
    final_jaccard=sum(jaccard_set)/70

fig = plt.figure()
fig.add_subplot(1, 2, 1)   
plt.imshow(binar_data,cmap="gray")
fig.add_subplot(1, 2, 2)
plt.imshow(binar_output,cmap="gray")

print(final_sensitivity)
