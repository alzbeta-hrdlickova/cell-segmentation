import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu

path_to_file='/Users/betyadamkova/Desktop/final/' 

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
    data=data[0,0,:,:]      
    #fig, ax = try_all_threshold(data, figsize=(10, 6), verbose=False)    #najít optimální threshold
    threshold1=threshold_otsu(data) 
    for x in range(224):      
        for y in range(224):
            if data[x,y] >= threshold1:
                data[x,y] = 1
            else:
                data[x,y] = 0
    #plt.imshow(data,cmap="gray")
    
    output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy')
    output=output[0,0,:,:]
    threshold2= threshold_otsu(output)
    for x in range(224):
        for y in range(224):
            if output[x,y] <= threshold2:
                output[x,y] = 0
            else:
                output[x,y] = 1
    #plt.imshow(output,cmap="gray")
          
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
plt.imshow(data,cmap="gray")
fig.add_subplot(1, 2, 2)
plt.imshow(output,cmap="gray")

print(final_sensitivity)
  
