""" Funkce SEG - Evaluation of segmentation accuracy """
def SEEGacc (img, gt_mask):
    """ img. gt_mask as np array of the same size
        classes - class of the object in the np array (fe 1))"""
    [row,col]=img.shape
    match=0
    classcount_gtmask=0
    classcount_img =0
    
    for r in range (0,row):
        for c in range (0,col):
            if gt_mask [r,c]>=1:
                classcount_gtmask=classcount_gtmask+1
                
            if img [r,c]>=1:
                classcount_img=classcount_img+1
    
    
    for r in range (0,row):
        for c in range (0,col):
            if img[r,c]>=1 and img[r,c]== gt_mask[r,c]:
                match=match+1


    if match>0.5*classcount_gtmask:
        acc=abs(match)/abs(((classcount_gtmask+classcount_img)-match))
    else:
        acc=0

    return acc
