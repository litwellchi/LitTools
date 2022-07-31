import numpy as np 
import cv2 

def show_result(img,
                results,
                bbox_color=(255, 101, 241),
                out_file=None):

    img = cv2.imread(img)
    added_image = img.copy()
    segm_result=results.f.arr_0
    # draw segmentation masks
    for mask in segm_result:
        contours, im = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image=added_image,contours=contours,contourIdx=-1,color=bbox_color,thickness=2)
    return added_image



img_dir = ''
mask_dir = 'crop.npz'
results = np.load(mask_dir)

img = show_result(img_dir,results)

cv2.imwrite('./testmask.jpg',img)