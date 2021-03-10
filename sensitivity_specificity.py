import numpy as np
import matplotlib.pyplot as plt

path_to_file='/Users/betyadamkova/Desktop/final/' 

sensitivity_set=[] 

it=-1
for fin in range(70):
    it+=1
    print(it)
    
    data =np.load('/Users/betyadamkova/Desktop/final/data/' + 'data' + str(it) +'.npy') 
    data=data[0,0,:,:]
    threshold1=np.mean(data)
    for x in range(224):      
        for y in range(224):
            if data[x,y] >= threshold1:
                data[x,y] = 1
            else:
                data[x,y] = 0
    #plt.imshow(data,cmap="gray")
    
    output =np.load('/Users/betyadamkova/Desktop/final/output/' + 'output' + str(it) +'.npy')
    output=output[0,0,:,:]
    threshold2=np.mean(output)
    for x in range(224):
        for y in range(224):
            if output[x,y] <= threshold2:
                output[x,y] = 0
            else:
                output[x,y] = 1
    plt.imshow(output,cmap="gray")
          
    TP = np.sum(((data==1) & (output ==1)).astype(np.float32))
    FN = np.sum(((data==0) & (output ==0)).astype(np.float32))
    sensitivity = TP / (TP + FN)
    sensitivity_set.append(sensitivity)
    final_sensitivity=sum(sensitivity_set)/70

fig = plt.figure()
fig.add_subplot(1, 2, 1)   
plt.imshow(data,cmap="gray")
fig.add_subplot(1, 2, 2)
plt.imshow(output,cmap="gray")

print(final_sensitivity)


#precision =  TP/ (FP + TP)
#accuracy = (TN + TP)/ (TN + FP + FN + TP)
#Dice = 2*TP / ( 2*TP + FP + FN )
#Jaccard = TP / ( TP + FP + FN ) 
