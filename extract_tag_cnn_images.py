#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:51:14 2018

@author: julien
"""

import cv2
import os
from python import darknet as dn


# load the Neural Network and the meta
path = "./char/"
cfg = os.path.join(path,"yolov3_plate.cfg").encode()
weights = os.path.join(path,"yolov3_plate_4000_ok.weights").encode()
data = os.path.join(path,"obj_plate.data").encode()

net = dn.load_net(cfg,weights, 0)
meta = dn.load_meta(data)

path_v = "/home/jehl/darknet/char/img_char"



files = os.listdir(path_v)

for f in files:
    img_path = os.path.join(path_v,f)
    result, im = dn.detect(net, meta, img_path.encode(), thresh=0.1)
    if len(result) !=0:
        print("Recognize !!!!!!!!!!!!!!!!!!")
        for p in result:
            x1 = int(p[2][0]-p[2][2]/2)
            y1 = int(p[2][1]-p[2][3]/2)
            x2 = int(p[2][0]+p[2][2]/2)
            y2 = int(p[2][1]+p[2][3]/2)
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            #cv2.putText(frame, p["plate"], (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            h, w = im.h, im.w
            center =  (p[2][0],p[2][1])
            hb, wb = (p[2][3],p[2][2])
            #cv2.circle(frame, center, 10, (255,0,0),2)
            #cv2.line(frame, (center[0]-wb/2,center[1]), (center[0]+wb/2,center[1]), (255,0,0),3)
            #cv2.line(frame, (center[0],center[1]-hb/2), (center[0],center[1]+hb/2), (255,0,0),3)
            
            f_name = f.split('.')[0]
            with open(f_name+'.txt', 'a+') as ftxt:
                l='0 '+str(float(center[0])/w)+" "+str(float(center[1])/h)+" "+str(float(wb)/w)+" "+str(float(hb)/h)+"\n"
                ftxt.write(l)
            
        

    else : 
        print("*")
            