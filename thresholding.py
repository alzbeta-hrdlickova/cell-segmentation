import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects

sensitivity_set=[] 
specificity_set=[]
accuracy_set=[]
dice_set=[]
jaccard_set=[]

it=-1
for fin in range(70):
        it+=1
    
        data =np.load('/Users/betyadamkova/Desktop/final/data/' + 'data' + str(it) +'.npy') 
        data=data[0,0,:,:]   
        data=data>0
        data=img_as_float(data)
        #plt.imshow(data,cmap="gray")
        
        output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy')
        output=output[0,0,:,:]
        #plt.imshow(output,cmap="gray")
        
        binar_output = output >-0.32
        #plt.imshow(binar_output,cmap="gray")
        binar_output = remove_small_holes(remove_small_objects(binar_output, 50),50)
        #plt.imshow(pokus,cmap="gray")
        
        output=img_as_float(binar_output)
        
        TP = np.sum(((data==1) & (output ==1)).astype(np.float64))      
        FN = np.sum(((data==1) & (output ==0)).astype(np.float64))      
        TN = np.sum(((data==0) & (output ==0)).astype(np.float64))
        FP = np.sum(((data==0) & (output ==1)).astype(np.float64))    
            
        sensitivity = TP / (TP + FN)        
        specificity = TN / (TN + FP)
        accuracy = (TN + TP)/ (TN + FP + FN + TP)
        Dice = 2*TP / ( 2*TP + FP + FN )
        Jaccard = TP / ( TP + FP + FN ) 
        
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

print("Výsledky :") 
print("Senzitivity =", final_sensitivity)
print("Specificita =", final_specificity)     
print("Accuracy =", final_accuracy)
print("Dice koeficient=", final_dice) 
print("Jaccard koeficient:", final_jaccard) 


fig, axes = plt.subplots(ncols=2, figsize=(15, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(data, cmap=plt.cm.gray)
ax[0].set_title('binar_data')
ax[0].axis('off')
ax[1].imshow(binar_output, cmap=plt.cm.nipy_spectral)
ax[1].set_title('binar_output')
ax[1].axis('off')
     
