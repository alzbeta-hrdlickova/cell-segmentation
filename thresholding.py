import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
from skimage.filters import threshold_yen
from skimage.filters import threshold_isodata

def binar(data):
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

otsu_sensitivity=0
otsu_specificity=0
otsu_accuracy=0
otsu_dice=0
otsu_jaccard=0

mean_sensitivity=0
mean_specificity=0
mean_accuracy=0
mean_dice=0
mean_jaccard=0

yen_sensitivity=0
yen_accuracy=0
yen_dice=0
yen_jaccard=0

isodata_sensitivity=0
isodata_accuracy=0
isodata_dice=0
isodata_jaccard=0

kk=-1
for k in range (4):
    kk+=1
    print(kk)
    sensitivity_set=[] 
    specificity_set=[]
    accuracy_set=[]
    dice_set=[]
    jaccard_set=[]
        
    it=-1
    for fin in range(70):
        it+=1
        #print(it)
    
        data =np.load('/Users/betyadamkova/Desktop/final/data/' + 'data' + str(it) +'.npy') 
        data=data[0,0,:,:]
        if kk == 1:
            threshold=threshold_otsu(data)
        elif kk ==2:
            threshold=threshold_mean(data)
        elif kk ==3:
            threshold=threshold_yen(data)
        else:
            threshold=threshold_isodata(data)
            
        binar_data=binar(data)
        
        output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy')
        output=output[0,0,:,:]
        if kk == 1:
            threshold=threshold_otsu(output)
        elif kk ==2:
            threshold=threshold_mean(output)
        elif kk ==3:
            threshold=threshold_yen(output)
        else:
            threshold=threshold_isodata(output)
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

        if kk == 1:
            otsu_sensitivity=final_sensitivity
            otsu_specificity=final_specificity
            otsu_accuracy=final_accuracy
            otsu_dice=final_dice
            otsu_jaccard=final_jaccard
        elif kk ==2:
            mean_sensitivity=final_sensitivity
            mean_specificity=final_specificity
            mean_accuracy=final_accuracy
            mean_dice=final_dice
            mean_jaccard=final_jaccard
        elif kk ==3:
            yen_sensitivity=final_sensitivity
            yen_specificity=final_specificity
            yen_accuracy=final_accuracy
            yen_dice=final_dice
            yen_jaccard=final_jaccard
        else:
            isodata_sensitivity=final_sensitivity
            isodata_specificity=final_specificity
            isodata_accuracy=final_accuracy
            isodata_dice=final_dice
            isodata_jaccard=final_jaccard
        

print("Výsledky senzitivity:")      
print("Sensitivita při Otsu threshold =", otsu_sensitivity,)
print("Sensitivita při Mean threshold =", mean_sensitivity)     
print("Sensitivita při Yen threshold =", yen_sensitivity)
print("Sensitivita při Isodata threshold =", isodata_sensitivity)
SS=(otsu_sensitivity + mean_sensitivity  + yen_sensitivity + isodata_sensitivity)/4
print("Průměr vypočtených senzitivit:", SS) 

print("Výsledky specificity:") 
print("Specificity při Otsu threshold =", otsu_specificity)
print("Specificity při Mean threshold =", mean_specificity)     
print("Specificity při Yen threshold =", yen_specificity)
print("Specificity při Isodata threshold =", isodata_specificity)
SP=(otsu_specificity + mean_specificity  + yen_specificity + isodata_specificity)/4
print("Průměr vypočtených specificit:", SP) 

print("Výsledky accuracy:") 
print("Accuracy při Otsu threshold =", otsu_accuracy)
print("Accuracy při Mean threshold =", mean_accuracy)     
print("Accuracy při Yen threshold =", yen_accuracy)
print("Accuracy při Isodata threshold =", isodata_accuracy)
Acc=(otsu_accuracy + mean_accuracy  + yen_accuracy + isodata_accuracy)/4
print("Průměr vypočtených accuracy:", Acc) 

print("Výsledky dice koeficientu:") 
print("Dice koeficient při Otsu threshold =", otsu_dice)
print("Dice koeficient při Mean threshold =", mean_dice)     
print("Dice koeficient při Yen threshold =", yen_dice)
print("Dice koeficient při Isodata threshold =", isodata_dice)
D=(otsu_dice + mean_dice  + yen_dice + isodata_dice)/4
print("Průměr vypočtených dice koeficientů:", D) 

print("Výsledky jaccard koeficientu:") 
print("Jaccard koeficient při Otsu threshold =", otsu_jaccard)
print("Jaccard koeficient při Mean threshold =", mean_jaccard)     
print("Jaccard koeficient při Yen threshold =", yen_jaccard)
print("Jaccard koeficient při Isodata threshold =", isodata_jaccard)
J=(otsu_jaccard + mean_jaccard  + yen_jaccard + isodata_jaccard)/4
print("Průměr vypočtených jaccard koeficientu:", J) 
      
