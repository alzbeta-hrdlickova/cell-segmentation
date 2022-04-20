import matplotlib.pyplot as plt
import numpy as np

#vytvoření datasetu
data = [sensitivity_set, specificity_set, accuracy_set, dice_set, jaccard_set]
 
fig = plt.figure(figsize =(10, 6))
ax = fig.add_subplot(111)

# instance
bp = ax.boxplot(data, patch_artist = True,notch ='True', vert = 1)
barva = ['green', 'yellow','pink', 'blue', 'orange']
 
for patch, color in zip(bp['boxes'], barva):patch.set_facecolor(color)

# změna vzhledu 
for whisker in bp['whiskers']:whisker.set(color ='gray',linewidth = 1.5,linestyle =":")
for cap in bp['caps']:cap.set(color ='gray',linewidth = 1)
for median in bp['medians']:median.set(color ='red',linewidth = 3)
for flier in bp['fliers']:flier.set(marker ='D', color ='gray', alpha = 0.5)
     
# y-axis názvy a nadpis
ax.set_xticklabels(['Senzitivita', 'Specificita','Accuracy', 'Jaccard', 'Dice koeficient'])
plt.title("Box plot výsledků")
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
     
plt.show()
