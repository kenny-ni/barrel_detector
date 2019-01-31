# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:18:59 2019

@author: jiageng
"""
import numpy as np
#from matplotlib import pyplot as plt
import cv2

blue_cr = np.array([])
blue_cb = np.array([])
other_cr = np.array([])
other_cb = np.array([])
for index in range(1,26):
    mask = np.load("C:/users/jiageng/Desktop/ECE276A_HW1/masks/%s.npy" % index)
    mask_inv = np.invert(mask)
    img = cv2.imread("C:/users/jiageng/Desktop/ECE276A_HW1/trainset/%s.png" % index)
    #change color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img_y = img[:,:,0]
    img_cr = img[:,:,1]
    img_cb = img[:,:,2]
    
    sample_blue_cr = np.extract(mask, img_cr)
    sample_blue_cb = np.extract(mask, img_cb)
    sample_other_cr = np.extract(mask_inv, img_cr)
    sample_other_br = np.extract(mask_inv, img_cb)
    blue_cr = np.append(blue_cr, sample_blue_cr)
    blue_cb = np.append(blue_cb, sample_blue_cb)
    other_cr = np.append(other_cr, sample_other_cr)
    other_cb = np.append(other_cb, sample_other_br)

#blue_mean_cr = np.mean(blue_cr)  
#blue_mean_cb = np.mean(blue_cb)
#other_mean_cr = np.mean(other_cr)
#other_mean_cb = np.mean(other_cb) 
    
#prior probability
py_blue = len(blue_cr)/(len(blue_cr) + len(other_cr))
py_other = len(other_cr)/(len(blue_cr) + len(other_cr))
blue = np.stack((blue_cr, blue_cb))
other = np.stack((other_cr, other_cb))
#calculate mean and covariance
#mean: 1*2
mean_blue = np.mean(blue, axis = 1)
mean_blue = mean_blue.reshape(2,1)
mean_other = np.mean(other, axis = 1)
mean_other = mean_other.reshape(2,1)
#convariance: 2*2
covariance_blue = np.cov(blue)
covariance_other = np.cov(other)

#save model parameters
np.save("C:/users/jiageng/Desktop/ECE276A_HW1/parameters/mean_blue.npy" , mean_blue)
np.save("C:/users/jiageng/Desktop/ECE276A_HW1/parameters/mean_other.npy" , mean_other)
np.save("C:/users/jiageng/Desktop/ECE276A_HW1/parameters/cov_blue.npy" , covariance_blue)
np.save("C:/users/jiageng/Desktop/ECE276A_HW1/parameters/cov_other.npy" , covariance_other)
np.save("C:/users/jiageng/Desktop/ECE276A_HW1/parameters/py_blue.npy", py_blue)
np.save("C:/users/jiageng/Desktop/ECE276A_HW1/parameters/py_other.npy", py_other)