# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 15:34:35 2018

@author: julien
"""
#              PART 1  ----------------- gen plate -------------------------------------------------------------------------------
import cv2
import random
import string
import numpy as np

#place = {1:(9,35),2:(9,67),3:(9,115),4:(9,147),5:(9,179),6:(9,227),7:(9,260),8:(3,291)}
place = {1:(13,35),2:(13,67),3:(13,115),4:(13,147),5:(13,179),6:(13,227),7:(13,260),8:(7,291)}

classes = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14
           ,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,'0':14,'1':26,
           '2':27,'3':28,'4':29,'5':30,'6':31,'7':32,'8':33,'9':34}

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),cv2.BORDER_CONSTANT, borderValue=(255,255,255)), M

def insert(char,nb,img):
    char = cv2.imread("plaque/police3/"+char+".jpg")
    img[place[nb][0]:place[nb][0]+char.shape[0],place[nb][1]:place[nb][1]+char.shape[1]]=char
    #cv2.rectangle(img, (place[nb][1],place[nb][0]), (place[nb][1]+char.shape[1],place[nb][0]+char.shape[0]), (0,255,0), 2)
    A = np.array([[[place[nb][1],place[nb][0]],[place[nb][1]+char.shape[1],place[nb][0]],[place[nb][1]+char.shape[1],place[nb][0]+char.shape[0]],[place[nb][1],place[nb][0]+char.shape[0]]]], dtype=np.float32)
    return A
    
img_vierge = cv2.imread("plaque/vierge4.jpg") 
plate = 20000
for nbp in range(10000):
    
    img=img_vierge.copy()
    i=1
    coord = []
    for _ in range(2):
        l = random.choice(string.ascii_letters[26:52])
        coord.append((l,insert (l,i,img)))
        i+=1
    for _ in range(3):
        n = random.choice(range(0,10))
       
        coord.append((n,insert (str(n),i,img)))
        i+=1 
    
    for _ in range(2):
        l = random.choice(string.ascii_letters[26:52])
        coord.append((l,insert (l,i,img)))
        i+=1
    '''   
    r = random.choice(('da','dd','dr','db'))
    insert(r,i,img)
    '''
    s = random.uniform(0.1, 3)
    img = cv2.resize(img,(0,0),fx=s, fy=s, interpolation = cv2.INTER_CUBIC)
    
    h,w = img.shape[:2]
    f1 , f2 = int(h/10), int(w/30)
    p=m=random.randint(-f1,f1)
    u=v=random.randint(-f2,f2)
    (w1,w2,h1,h2)= (w*1/50, w*49/50, h*1/50, h*49/50) 
    pts1 = np.float32([[w1,h1],[w2,h1],[w2,h2],[w1,h2]])
    pts2 = np.float32([[w1-u,h1-m],[w2+u,h1-p],[w2-v,h2+p],[w1+v,h2+m]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    
    img = cv2.warpPerspective(img,M,(w,h), cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    angle = random.randint(-10,10)
    img, M2 = rotate_bound(img,angle)
    h,w = img.shape[:2]
    
    new_coord=[]
    for letter, box in coord:
        mat = cv2.perspectiveTransform(box*s, M)
        mat = cv2.transform(mat,M2)
        miny=min(mat[0][:,1])
        maxy =max(mat[0][:,1])
        minx=min(mat[0][:,0])
        maxx =max(mat[0][:,0])
        new_coord.append((letter,(((minx+maxx)/2/w,(miny+maxy)/2/h),((maxx-minx)/w,(maxy-miny)/h))))
    
    
    img2=img.copy()
    m = (0,0,0) 
    s = (10,10,10)
    cv2.randn(img2,m,s);
    img = img + img2
    cv2.imwrite("plaque/plate/"+('00000'+str(plate))[-5:]+".jpg",img)
    with open("plaque/plate/"+('00000'+str(plate))[-5:]+".txt", 'w+') as f:
        for line in new_coord :
            f.write(' '.join((str(classes[(str(line[0]))]),str(line[1][0][0]),str(line[1][0][1]),str(line[1][1][0]),str(line[1][1][1])))+"\n")
    plate+=1
    print(plate)


# ------------------------------------- PART 2 TEST perspective -------------------------------------------------------------------
from PIL import Image

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img_rgb)
im_pil
img2=img.copy()
m = (0,0,0) 
s = (10,10,10)
cv2.randn(img2,m,s);

img3 = img+img2

p=10
m=-10
cv2.perspectiveTransform(A, M)
A = np.array([[[35, 13],[62,57]]], dtype=np.float32)
for center in new_coord :
    cv2.circle(img,(int(center[1][0][0]),int(center[1][0][1])),5,(255,0,0),2)
    print(center)

mat = cv2.perspectiveTransform(coord[0], M)
new_coord
coord


# PART 3         Gen the police in differetn colors -------------------------------------------------------------------------------

#!/usr/local/bin/python3
import cv2 as cv
import numpy as np
import os

path_v = "./plaque/police/"

files = os.listdir(path_v)

for f in files:
    
    # Load the aerial image and convert to HSV colourspace
    image = cv.imread(path_v+f)
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)

    
    # Define lower and uppper limits of what we call "brown"
    brown_lo=np.array([0,0,0])
    brown_hi=np.array([255,255,180])
    
    mask=cv.inRange(hsv,brown_lo,brown_hi)
    
    # Change image to red where we found brown
    image[mask>0]=(0,0,180)
    
    
    brown_lo=np.array([0,0,181])
    brown_hi=np.array([255,255,255])
    
    # Mask image to only select browns
    mask=cv.inRange(hsv,brown_lo,brown_hi)
    
    # Change image to red where we found brown
    image[mask>0]=(0,0,15)
    
    image = cv.cvtColor(image,cv.COLOR_HSV2BGR)
    
    cv.imwrite("plaque/police3/"+f,image)
