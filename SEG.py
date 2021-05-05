""" Funkce SEG - Evaluation of segmentation accuracy """
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float

def SEEGacc (wat, maska):

    [row,col]=maska.shape
    match=0
    acc=0
    JaccIn=[]
    jaccard=[]
    pocet_bunek= (np.amax(maska)).astype(np.int)         #počet buněk
    pocet_bunek2=np.amax(wat)
    pomoc=maska
    pomoc2=wat
    
    for r in range (1,pocet_bunek+1):     
        for c in range (1,pocet_bunek2+1): 
            #print('r:',r)

            maska=(maska!=r)==0              #nahrazení ostatních buněk nulou
            #plt.imshow(maska,cmap="gray")
            maska=img_as_float(maska)       #jedna bunka oznacena 1
            pocet1=np.sum(maska)            #počet pixelu bunky
                
            wat=(wat!=c)==0 
            wat=img_as_float(wat)
            #plt.imshow(wat,cmap="gray")
            pocet2=np.sum(wat)
            
            for k in range (0,row):
                for l in range (0,col):
                    if maska[k,l]== 1 and maska[k,l] == wat[k,l]:
                        match=match+1
                        
                        if match>0.5*pocet2:
                            acc=abs(match)/abs(((pocet1+pocet2)-match))
                        else:
                            acc=0
                            
            JaccIn.append(acc)
            maska=pomoc
            wat=pomoc2
            match=0
            acc=0
            
        jaccard.append(max(JaccIn))
        #print(max(JaccIn) )
        JaccIn=[]
        
    return sum(jaccard)/(pocet_bunek)
