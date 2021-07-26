import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects

'''prahování predikovaného snímku, výpočet hodnot Senzitivity, Specificity, Accuracy, Dice koeficientu a Jaccard koeficientu
    nastavení prahů, výpočet hodnot a získání nejlepších výsledků dle Jaccard koeficinetu  '''
    
prah1=1.2
prah2=1.3
prah3=1.4

sensitivity_set=[] 
specificity_set=[]
accuracy_set=[]
dice_set=[]
jaccard_set=[]

for kk in range(3):

    sensitivity_set=[] 
    specificity_set=[]
    accuracy_set=[]
    dice_set=[]
    jaccard_set=[]
    
    it=-1
    for fin in range(70):
        it+=1
        
        data =np.load('/Users/betyadamkova/Desktop/final/model 8/lbl/' + 'lbl' + str(it) +'.npy') 
        data=data[0,0,:,:]   
        data=data>1
        data=img_as_float(data)                          #binární maska
        #plt.imshow(data,cmap="gray")
        
        output =np.load('/Users/betyadamkova/Desktop/final/model 8/output/' + 'output' + str(it) +'.npy')
        binar_output=output[0,0,:,:]
        #plt.imshow(binar_output,cmap="gray")
        
        if kk ==0:
            binar_output = binar_output>prah1
        elif kk==1:
            binar_output = binar_output>prah2
        elif kk ==2:
            binar_output = binar_output>prah3
        
        binar_output = remove_small_holes(remove_small_objects(binar_output, 15),1200)
        output=img_as_float(binar_output)                    #binární predikovaný obraz
        
        TP = np.sum(((data==1) & (output ==1)).astype(np.float64))     
        FN = np.sum(((data==1) & (output ==0)).astype(np.float64))      
        TN = np.sum(((data==0) & (output ==0)).astype(np.float64))
        FP = np.sum(((data==0) & (output ==1)).astype(np.float64))    
             
        if (TP + FN) ==0:                                      #výpočet metrik, zajištění výpočtu při prázdném obrazu
            sensitivity=0
        else:
            sensitivity = TP / (TP + FN)   
            
        specificity = TN / (TN + FP)
        accuracy = (TN + TP)/ (TN + FP + FN + TP)
        
        if ( 2*TP + FP + FN )==0:
            Dice= 0
        else:
            Dice = 2*TP / ( 2*TP + FP + FN )
            
        if ( TP + FP + FN )  == 0:
            Jaccard =0
        else:
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
        
        if kk ==0:
             prah1_sensitivity = final_sensitivity
             prah1_specificity=final_specificity
             prah1_accuracy=final_accuracy
             prah1_dice=final_dice
             prah1_jaccard=final_jaccard
        elif kk ==1:
             prah2_sensitivity=final_sensitivity
             prah2_specificity=final_specificity
             prah2_accuracy=final_accuracy
             prah2_dice=final_dice
             prah2_jaccard=final_jaccard
        elif kk ==2:
             prah3_sensitivity=final_sensitivity
             prah3_specificity=final_specificity
             prah3_accuracy=final_accuracy
             prah3_dice=final_dice
             prah3_jaccard=final_jaccard

if prah1_jaccard == np.amax([prah1_jaccard, prah2_jaccard, prah3_jaccard]):
    print("Práh=:", prah1)
    print("Senzitivita=:", prah1_sensitivity) 
    print("Specificita=:", prah1_specificity)
    print("Accuracy=:", prah1_accuracy)
    print("Dice koeficient=:", prah1_dice)
    print("Jaccard koeficient=:", prah1_jaccard)
elif prah2_jaccard == np.amax([prah1_jaccard, prah2_jaccard, prah3_jaccard]):
    print("Práh=:", prah2)
    print("Senzitivita=:", prah2_sensitivity) 
    print("Specificita=:", prah2_specificity)
    print("Accuracy=:", prah2_accuracy)
    print("Dice koeficient=:", prah2_dice)
    print("Jaccard koeficient=:", prah2_jaccard)
elif prah3_jaccard == np.amax([prah1_jaccard, prah2_jaccard, prah3_jaccard]):
    print("Práh=:", prah3)
    print("Senzitivita=:", prah3_sensitivity) 
    print("Specificita=:", prah3_specificity)
    print("Accuracy=:", prah3_accuracy)
    print("Dice koeficient=:", prah3_dice)
    print("Jaccard koeficient=:", prah3_jaccard) 


fig, axes = plt.subplots(ncols=2, figsize=(15, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(data, cmap=plt.cm.gray)
ax[0].set_title('binární maska')
ax[0].axis('off')
ax[1].imshow(output, cmap=plt.cm.gray)
ax[1].set_title('binaární predikovaný snímek')
ax[1].axis('off')
