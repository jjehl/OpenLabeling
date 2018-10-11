# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 08:15:38 2018

@author: julien
"""
import cv2

for i in range(15000):
    i+=1
    f = ('0000'+str(i))[-5:]
    #print(f)
    img = cv2.imread('./plate_v2/'+f+'.jpg')
    if img is not None :
        #print('get img')
        h, w=img.shape[:2]
        with open('./plate_v2/'+f+'.txt', 'r') as ft:
            nl = 1
            for l in ft:
                #print('get plate {}'.format(nl))
                nums = [float(n) for n in l.split()]
                r = (int(nums[1]*w), int(nums[2]*h), int(nums[3]*w), int(nums[4]*h))
                imgr=img[max(r[1]-r[3]//2,0):max(r[1]+r[3]//2,0), max(r[0]-r[2]//2,0):max(r[0]+r[2]//2,0)]
                cv2.imwrite('./char/'+f+'_'+str(nl)+'.jpg', imgr)
                nl += 1

"""
from PIL import Image

img_rgb = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img_rgb)
im_pil
"""
