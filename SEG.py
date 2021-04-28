
def SEEGacc (img, gt_mask, classes):
    """ hodnocení kvality segmentace založené na Jaccard similarity index, zdroj informací https://public.celltrackingchallenge.net/documents/SEG.pdf
        img. gt_mask v numpy array o stejné velikosti
        classes - třída objektu-pozadi v np array (0,1)"""
    [row,col]=img.shape
    match=0
    classcount_gtmask = (gt_mask == classes).sum()
    classcount_img = (img == classes).sum()
    for r in range (0,row):
        for c in range (0,col):
            if img[r,c]==gt_mask[r,c]:
                match=match+1

    if ~(classcount_gtmask>0.5*match):
        acc=0
    else:
        acc=match/((classcount_gtmask+classcount_img)-match)

    return acc
