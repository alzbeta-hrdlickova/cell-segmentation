import numpy as np
import matplotlib.pyplot as plt

path_to_file='images/final/' 

sensitivity_set=[] 

it=-1
for fin in range(70):
    it+=1
    print(it)
    
    lbl =np.load(path_to_file + 'lbl' + str(it) +'.npy') 
    lbl=lbl[0,0,:,:]
    threshold1=np.mean(lbl)
    for x in range(224):      
        for y in range(224):
            if lbl[x,y] >= threshold1:
                lbl[x,y] = 1
            else:
                lbl[x,y] = 0
    plt.imshow(lbl,cmap="gray")
         
    output =np.load(path_to_file + 'output' + str(it) +'.npy')
    output=output[0,0,:,:]
    threshold2=np.mean(output)
    for x in range(224):
        for y in range(224):
            if output[x,y] <= threshold2:
                output[x,y] = 0
            else:
                output[x,y] = 1
                plt.imshow(output,cmap="gray")
          
    TP = np.sum(((lbl==1) & (output ==1)).astype(np.float32))
    FN = np.sum(((lbl==0) & (output ==0)).astype(np.float32))
    sensitivity = TP / (TP + FN)
    sensitivity_set.append(sensitivity)
    final_sensitivity=sum(sensitivity_set)/70

fig = plt.figure()
fig.add_subplot(1, 2, 1)   
plt.imshow(lbl,cmap="gray")
fig.add_subplot(1, 2, 2)
plt.imshow(output,cmap="gray")

print(final_sensitivity)
