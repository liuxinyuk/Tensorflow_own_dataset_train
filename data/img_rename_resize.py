# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:04:24 2017
@author: wuhui
用来对图片进行重命名，并初始化目录结构
"""

import cv2
import os

if __name__=='__main__':
    path = os.getcwd()
    imgs = os.listdir(path+"\\images\\")
    if os.path.exists('JPEGImages') == False:
        os.mkdir('JPEGImages')
    if os.path.exists('Annotations') == False:
        os.mkdir('Annotations')
    cnt = 1
    prename = "000000"
    for img in imgs:
        temp=cv2.imread(path+"\\images\\"+img)
        #os.remove(path+"\\images\\"+img)
        temp=cv2.resize(temp,(500,500),cv2.INTER_AREA) #重采样插值法
        cv2.imwrite(path+"\\JPEGImages\\"+prename[0:len(prename)-len(str(cnt))]+str(cnt)+".jpg",temp)
        print "renamed "+img+" to "+prename[0:len(prename)-len(str(cnt))]+str(cnt)+".jpg"
        cnt+=1
    print 'done!'
