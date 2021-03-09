import numpy as np

path_to_file='images/final/'

it=-1
for fin in range(1):
    it+=1

    lbl =np.load(path_to_file + 'lbl' + str(it) +'.npy') 
    for x in range(lbl[0,0,:,:].shape[0]):
        a=lbl[0,0,0,:]
        a1=np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
        sum(a1)
        for y in range(lbl[0,0,:,:].shape[1]):
                    aa=lbl[0,0,:,0]
                    a2=np.r_[True, aa[1:] < aa[:-1]] & np.r_[aa[:-1] < aa[1:], True]
                    sum(a2)
                    
    output =np.load(path_to_file + 'output' + str(it) +'.npy')
    for x in range(output[0,0,:,:].shape[0]):
        b=output[0,0,0,:]
        b1= np.r_[True, b[1:] < b[:-1]] & np.r_[b[:-1] < b[1:], True]
        sum(b1)
        for y in range(output[0,0,:,:].shape[1]):
                    bb=output[0,0,:,0]
                    b2=np.r_[True, bb[1:] < bb[:-1]] & np.r_[bb[:-1] < bb[1:], True]
                    sum(b2)

                    
