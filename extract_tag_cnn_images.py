#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:51:14 2018

@author: julien
"""

import cv2
import os
import darknet as dn


# load the Neural Network and the meta
path = "./"
cfg = os.path.join(path,"yolov3.cfg").encode()
weights = os.path.join(path,"yolov3_2300.weights").encode()
data = os.path.join(path,"obj.data").encode()

net = dn.load_net(cfg,weights, 0)
meta = dn.load_meta(data)

path_v = '/home/julien/Videos'



files = os.listdir(path_v)
i=1
for f in files:
    vcap = cv2.VideoCapture(path_v+"/"+f)

    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret :
           print('end') 
           break
        custom_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #custom_image = cv2.resize(custom_image,(dn.lib.network_width(net), dn.lib.network_height(net)), interpolation = cv2.INTER_LINEAR)            
        image, arr = dn.array_to_image(custom_image)
        result = dn.detect(net, meta, image, thresh=0.2)
        if len(result) !=0:
            print("Recognize !!!!!!!!!!!!!!!!!!")
            for p in result:
                x1 = int(p[2][0]-p[2][2]/2)
                y1 = int(p[2][1]-p[2][3]/2)
                x2 = int(p[2][0]+p[2][2]/2)
                y2 = int(p[2][1]+p[2][3]/2)
                #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                #cv2.putText(frame, p["plate"], (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                h, w = frame.shape[:2]
                center =  (p[2][0],p[2][1])
                hb, wb = (p[2][3],p[2][2])
                #cv2.circle(frame, center, 10, (255,0,0),2)
                #cv2.line(frame, (center[0]-wb/2,center[1]), (center[0]+wb/2,center[1]), (255,0,0),3)
                #cv2.line(frame, (center[0],center[1]-hb/2), (center[0],center[1]+hb/2), (255,0,0),3)
                
                with open("/home/julien/Pictures/"+str(i)+'.txt', 'a+') as f:
                    l='0 '+str(float(center[0])/w)+" "+str(float(center[1])/h)+" "+str(float(wb)/w)+" "+str(float(hb)/h)+"\n"
                    f.write(l)
                
            cv2.imwrite('/home/julien/Pictures/'+str(i)+'.jpg',frame)
            i+=1
    
        else : 
            print("*")
            
frame = cv2.imread("eu-11.jpg")
