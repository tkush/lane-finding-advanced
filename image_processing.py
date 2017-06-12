import numpy as np
import cv2
import matplotlib.pyplot as plt

# This class contains static methods used for processing
# each input image frame
class Image_Processing(object):
    
    @staticmethod
    # Sharpen the image to accentuate edges
    def sharpen_image(img):
        kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
        output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
        output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
        
        return output_2
    
    @staticmethod
    # Threshold the image based on a range
    def color_thres(img, thres=[0,255]):
        thresholded = np.zeros_like(img)
        thresholded[(img >= thres[0]) & (img <= thres[1])] = 1
        return thresholded

    @staticmethod
    # Threshold the input frame image
    def thresholded_img(img):
        sharp_img = Image_Processing.sharpen_image(img)
        
        # Detecting white lane lines
        hls = cv2.cvtColor(sharp_img,cv2.COLOR_BGR2HLS)
        l_thres = Image_Processing.color_thres(hls[:,:,1],[190,255])

        # Detecting yellow lane lines
        ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        cb_thres = Image_Processing.color_thres(ycrcb[:,:,2],[65,110])
        
        # Create a binary thresholded image
        combined = np.zeros_like(l_thres)
        combined[(cb_thres==1) | (l_thres==1)] = 1
        
        return combined