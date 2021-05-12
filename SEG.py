import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float

""" Funkce SEG - Evaluation of segmentation accuracy 
    výpočet podobnosti buňky v masce s každou buňkou v segmentovaném obraze, zachování největší podobnosti, 
    vždy jen jedna, díky pravidlu, že shoda je větší jak polovina porovnávané buňky"""

def SEEGacc (wat, maska):
    [row,col]=maska.shape
    match=0
    acc=0
    JaccIn=[]
    jaccard=[]
    pocet_bunek= (np.amax(maska)).astype(np.int)        #počet buněk - nejvyšší očíslovaná bunka
    pocet_bunek2=(np.amax(wat)).astype(np.int)
    pomoc=maska
    pomoc2=wat
    
    for r in range (1,pocet_bunek+1):  
        for c in range (1,pocet_bunek2 +1): 

            maska=(maska!=r)==0              #nahrazení ostatních buněk nulou
            maska=img_as_float(maska)        #jedna buňka oznacena 1
            #plt.imshow(maska,cmap="gray")
            pocet1=np.sum(maska)            #počet pixelu buňky
            
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
        JaccIn=[]
        
    if pocet_bunek ==0:             #pokud na snímku není žádná buňka, výsledek SEG=0
        return 0
    else: 
        return sum(jaccard)/(pocet_bunek)
