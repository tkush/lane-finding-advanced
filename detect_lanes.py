from frame_lane import *
from camera_calibration import *
from moviepy.editor import VideoFileClip
from frame_lane import *
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Calculate distortion data and store on disk
calculate_dist_coefficients('camera_cal','camera_cal')

# Get distortion data
coeffs = get_dist_coeffs('camera_cal/dist_coeff.p')

# Instantiate frame class
# To get the debugging video output, set the third arg
# to True:
# img_frame = Frame(coeffs,True,True)
img_frame = Frame(coeffs,True,True)

# # Get individual frame from the project video
# # Process frame and write to disk
# vidcap = cv2.VideoCapture('project_video.mp4')
# success,image = vidcap.read()
# success = True
# count = 1
# while success:
#     success,image = vidcap.read()
#     lanes = img_frame.process_frame(image)
#     file = 'non_debug/frame{:04d}.jpg'.format(count)
#     cv2.imwrite(file,lanes)
#     # cv2.imshow("",lanes)
#     # cv2.waitKey(5)
#     count += 1
# exit()

# Once the individual frames are available on disk
# create the video from the frames
images = []
for i in range(1,1253):
    images.append("frame{:04d}.jpg".format(i))

# Determine the width and height from the first image
image_path = os.path.join('non_debug', images[0])
frame = cv2.imread(image_path)
height, width, channels = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('out.mp4', fourcc, 15, (width, height))

for image in images:
    image_path = os.path.join('non_debug', image)
    frame = cv2.imread(image_path)
    out.write(frame) # Write out frame to video

out.release()
exit()
