# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:25:51 2019

@author: jiageng
"""

'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
#from roipoly import RoiPoly
#from matplotlib import pyplot as plt
import numpy as np

class BarrelDetector():
    def __init__(self):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
        '''
        
        self.mean_blue = np.load('mean_blue.npy')
        self.mean_other = np.load('mean_other.npy')
        self.cov_blue = np.load('cov_blue.npy')
        self.cov_other = np.load('cov_other.npy')
        self.py_blue = np.load('py_blue.npy')
        self.py_other = np.load('py_other.npy')


    def segment_image(self, img):
        '''
            Calculate the segmented image using a classifier
            eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        '''
        # YOUR CODE HERE
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        cr = img[:,:,1]
        height, width = cr.shape[0], cr.shape[1]
        number_of_pixel = cr.shape[0] * cr.shape[1]
        cr = np.reshape(cr,number_of_pixel)
        cb = img[:,:,2]
        cb = np.reshape(cb, number_of_pixel)
        pixel = np.stack((cr,cb))
        mask_img = self.multigausian(pixel,self.mean_blue, self.cov_blue, self.py_blue) < self.multigausian(pixel, self.mean_other, self.cov_other, self.py_other)
        mask_img = mask_img.reshape(height,width)
        return mask_img
    
    def multigausian(self, pixel, mean, cov, prior):
        temp = np.dot(np.transpose(pixel - mean) , cov**(-1)) * np.transpose((pixel - mean))
        temp = np.sum(temp, axis = 1) + np.log(np.linalg.det(cov)) - 2 * np.log(prior)
        return temp
    
    def  get_bounding_box(self, img):
        '''
            Find the bounding box of the blue barrel
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        mask_img = self.segment_image(img)
#        height = mask_img.shape[0]
        label_img = label(mask_img)
        regions = regionprops(label_img)
#         fig, ax = plt.subplots()
#         ax.imshow(label_img, cmap=plt.cm.gray)
        boxes = []
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            if maxc - minc < 20 or maxr - minr < 40 or (maxc - minc) / (maxr - minr) < 0.35 or (maxc - minc) / (maxr - minr) > 0.75 :
                continue
#            if maxc - minc < 10 or maxr - minr < 10 or (maxc - minc) / (maxr - minr) < 0.4 or (maxc - minc) / (maxr - minr) > 0.6:
#                continue
#             bx = (minc, maxc, maxc, minc, minc)
#             by = (minr, minr, maxr, maxr, minr)
            boxes.append([minc, minr, maxc, maxr])
#             ax.plot(bx, by, '-b', linewidth=2.5)
#         plt.show()
        return boxes


if __name__ == '__main__':

    folder = "trainset"
    my_detector = BarrelDetector()
#    img = cv2.imread("43.png")
#    box = my_detector.get_bounding_box(img)
#    print(box)
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
        mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
        boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope
