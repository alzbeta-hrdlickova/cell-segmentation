
def SEEGacc (img, gt_mask, class):
    """ img. gt_mask as np array of the same size
        classes - class of the object in the np array (fe 1))"""
    [row,col]=img.shape
    match=0
    classcount_gtmask = (gt_mask == class).sum()
    classcount_img = (img == class).sum()
    for r in range (0,row):
        for c in range (0,col):
            if img[r,c]==gt_mask[r,c]:
                match=match+1

    if ~(classcount_gtmask>0.5*match):
        acc=0
    else:
        acc=match/((classcount_gtmask+classcount_img)-match)

    return acc
