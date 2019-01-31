# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
from roipoly import RoiPoly
import matplotlib.image as mpimg

for index in range(1,26):
     # load image
     img = mpimg.imread("C:/users/jiageng/Desktop/ECE276A_HW1/trainset/%s.png" % index)
     fig = plt.figure()
     plt.imshow(img)
     plt.show(block=False)
     # draw the roi from the image
     roi = RoiPoly(color='b', fig=fig)
     mask_roi = roi.get_mask(img[:,:,1])
     plt.imshow(mask_roi,interpolation='nearest', cmap="Greys")
     plt.show()
     #store masks
     np.save("C:/users/jiageng/Desktop/ECE276A_HW1/masks/%s.npy" % index, mask_roi)
     print(index)