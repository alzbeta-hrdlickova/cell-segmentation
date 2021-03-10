import numpy as np
import matplotlib.pyplot as plt

path_to_file='/Users/betyadamkova/Desktop/final/' 

sensitivity_set=[] 

it=-1
for fin in range(70):
    it+=1
    print(it)
    
    """
    lbl =np.load('/Users/betyadamkova/Desktop/final/lbl/' + 'lbl' + str(it) +'.npy') 
    lbl=lbl[0,0,:,:]
    threshold1=np.mean(lbl)
    
    binar1=[]
    binar2=[]
    if threshold1 > -0.1: #bílý pozadí
        binar1 = 0  
        binar2 = 1  
    elif threshold1 < -0.1: #čený pozadí 
        binar1 = 1 
        binar2 = 0   
                 
    for x in range(224):      
        for y in range(224):
            if lbl[x,y] >= threshold1:
                lbl[x,y] = binar1
            else:
                lbl[x,y] = binar2
    #plt.imshow(lbl,cmap="gray")
    """
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
