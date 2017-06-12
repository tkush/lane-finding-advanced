# TO-DO: 
# Measure in non-warp space

import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_processing import *
import time
from scipy.optimize import curve_fit

# This class holds information for the two lanes
class Lane():
    def __init__(self,num):
        self.radius = -1
        self.fit_coeff = None
        self.prev_fit_coeff = None
        self.average_over_n_frames = num
        self.cur_x_cloud = None
        self.cur_y_cloud = None
        self.cur_radius = 0.0
        self.last_n_x_cloud = []
        self.last_n_y_cloud = []
        self.cur_x_points = []
        self.cur_y_points = []
        self.last_n_radius = []
        self.good_lane = False
        self.good_lane_tol = 0.30
        self.skip_count = 0

        # Define conversions in x and y from pixels space to meters
        self.ym_pp = 30/720 # meters per pixel in y dimension
        self.xm_pp = 3.7/700 # meters per pixel in x dimension
    
    # Function to calculate the radius of curvature (ROC)
    def calculate_roc(self, y_eval):
        y_float = np.asarray([float(y) for y in self.cur_y_points])
        x_float = np.asarray([float(x) for x in self.cur_x_points])
        fit_coefs_m = np.polyfit(y_float*self.ym_pp, x_float*self.xm_pp, 2)
        self.cur_radius = ((1 + (2*fit_coefs_m[0]*y_eval*self.ym_pp + fit_coefs_m[1])**2)**1.5) / np.absolute(2*fit_coefs_m[0])

    # Function to calculate whether the current lane detection is robust or not
    def is_lane_good(self):
        if ( len(self.last_n_radius)>0 ):
            avg_rad = sum(self.last_n_radius)/len(self.last_n_radius)
            if ( (abs(self.cur_radius - avg_rad)/avg_rad < self.good_lane_tol) ):
                self.good_lane = True
            else:
                self.good_lane = False
        else:
            # This is the first time detecting a lane, no previous data is available
            self.good_lane = True
    
    # Function for the optimization fit
    def polynomial_function(self, y, a, b, c):
        output = []
        for i in range(len(y)):
            output.append(int(a*y[i]*y[i] + b*y[i] + c))
        return output
    
    # Function to get single points from the point cloud
    def get_single_points(self,cur_only=True):
        self.cur_y_points = []
        self.cur_x_points = []
        if ( cur_only ):
            x_cloud_list_unsorted = self.cur_x_cloud.tolist()
            y_cloud_list_unsorted = self.cur_y_cloud.tolist()
        else:
            x_cloud_list_unsorted = np.concatenate(self.last_n_x_cloud)
            x_cloud_list_unsorted = np.reshape(x_cloud_list_unsorted,-1)
            y_cloud_list_unsorted = np.concatenate(self.last_n_y_cloud)
            y_cloud_list_unsorted = np.reshape(y_cloud_list_unsorted,-1)
        for_sorting = zip(y_cloud_list_unsorted,x_cloud_list_unsorted)
        sorted_zip_list = sorted(for_sorting)
        x_cloud_list = [x[1] for x in sorted_zip_list]
        y_cloud_list = [x[0] for x in sorted_zip_list]
        
        count = 0
        i = 0
        # This assumes that the list of x and y cloud points
        # are sorted according to the y points and in ascending
        # order
        while( i < len(x_cloud_list) - 1 ):
            y = y_cloud_list[i]
            x = []
            for j in range(i,len(x_cloud_list)):
                if ( y_cloud_list[j] == y ):
                    x.append(x_cloud_list[j])
                else:
                    self.cur_y_points.append(y)
                    self.cur_x_points.append(int(sum(x)/len(x)))
                    break
            i = j                
        
        for_sorting = zip(self.cur_y_points,self.cur_x_points)
        sorted_zip_list = sorted(for_sorting)
        self.cur_x_points = [x[1] for x in sorted_zip_list]
        self.cur_y_points = [x[0] for x in sorted_zip_list]

    # Function to fit second order polynomial through the points
    def get_poly_coeffs(self,fit_cur_only=True):
        if ( fit_cur_only ):
            self.fit_coeff = np.polyfit(self.cur_y_points, self.cur_x_points, 2)
        else:
            l_x_cloud = np.concatenate(self.last_n_x_cloud)
            l_x_cloud = np.reshape(l_x_cloud,-1)
            l_y_cloud = np.concatenate(self.last_n_y_cloud)
            l_y_cloud = np.reshape(l_y_cloud,-1)
            self.fit_coeff = np.polyfit(l_y_cloud, l_x_cloud, 2)
    
    def print_diagnostics(self, text):
        print("\n")
        print(text)
        print("Current cloud shape:\tX:{0}, Y:{1}".format(len(self.cur_x_cloud), len(self.cur_y_cloud)))
        print("History cloud shape:\tX:{0}, Y:{1}".format(len(self.last_n_x_cloud), len(self.last_n_y_cloud)))
        for i in range ( len (self.last_n_x_cloud) ):
            print("\tCloud {0}:\tX:{1}, Y:{2}".format(i+1,\
                                                      len(self.last_n_x_cloud[i]),\
                                                      len(self.last_n_y_cloud[i])))
        
        print("Current point shape:\tX:{0}, Y:{1}".format(len(self.cur_x_points), len(self.cur_y_points)))



       

# This class is used for each frame 
class Frame():
    def __init__(self, coefs, is_video=True, debug=False):
        
        # was the line detected in the last iteration?
        self.lane_detected = False
        
        # is the input a continuos video i.e. two consecutive
        # frames are highly co-related?
        self.is_video = is_video
        
        # Number of instances to keep in history
        self.num = 5

        # Lane classes
        self.left_lane = Lane(1)
        self.right_lane = Lane(3)
        
        # Distortion coefficeints
        self.mtx = coefs["mtx"]
        self.dist = coefs["dist"]

        # Produce a debug output?
        self.debug = debug

        # Keep count 
        self.count = 0

    # Take a camera image and draw lane lines on it
    # or if debug is True, then produce a frame with 
    # additional debug info on each frame
    def process_frame(self, frame):     
        
        # Define source and destination points for perspective transform
        src_points = np.float32([[220,700],[590,450],[690,450],[1090,700]])
        dst_points = np.float32([[300,720],[300,0],[800,0],[800,720]])
        
        # Get perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        
        # Undistort the frame based on the distortion coefficients calculated
        undist = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)   

        # Perspective transform
        img_size = (undist.shape[1], undist.shape[0])
        warped = cv2.warpPerspective(undist, M, img_size)
        
        # Threshold
        thresholded = Image_Processing.thresholded_img(warped)
        thresholded *= 255
        thresholded_stack = np.dstack((thresholded,thresholded,thresholded))
        # Create an output image to draw on and visualize the result
        out_img = np.zeros_like(thresholded_stack)
        warped_binary = thresholded
        
        if ( self.count < self.num ):
            margin = 75
            # print("Full search for left and right lanes... ")
            left_fitx, right_fitx, ploty = self.find_lanes_search_all(warped_binary,margin,'ALL')
        else: 
            # Left Lane
            if ( self.left_lane.skip_count < 5 ):
                margin = 75
                # print("Targeted search for left lane...")
                left_fitx, ploty = self.find_lanes_search_targeted(warped_binary,margin,'LEFT')
            else:
                margin = 50
                # print("Full search for left lane...")
                left_fitx, ploty = self.find_lanes_search_all(warped_binary,margin,'LEFT')
                
            # Right Lane
            if ( self.right_lane.skip_count < 5 ):
                margin = 100
                # print("Targeted search for right lane...")
                right_fitx, ploty = self.find_lanes_search_targeted(warped_binary,margin,'RIGHT')
            else:
                margin = 100
                # print("Full search for right lane...")
                right_fitx, ploty = self.find_lanes_search_all(warped_binary,margin,'RIGHT')
                
        # self.left_lane.print_diagnostics("Left Lane")
        # self.right_lane.print_diagnostics("Right Lane")

        thresholded_stack[self.left_lane.cur_y_cloud, self.left_lane.cur_x_cloud] = [255, 0, 0]
        thresholded_stack[self.right_lane.cur_y_cloud, self.right_lane.cur_x_cloud] = [0, 0, 255]
        for i in range(len(self.left_lane.cur_y_points)):
            cv2.circle(out_img,(int(self.left_lane.cur_x_points[i]),int(self.left_lane.cur_y_points[i])),2,(0,255,0))
        for i in range(len(self.right_lane.cur_y_points)):
            cv2.circle(out_img,(int(self.right_lane.cur_x_points[i]),int(self.right_lane.cur_y_points[i])),2,(0,255,0))

        pts_l_fit = []
        pts_r_fit = []
        for i in range(len(left_fitx)):
            pts_l_fit.append([left_fitx[i], ploty[i]])
        for i in range(len(right_fitx)):
            pts_r_fit.append([right_fitx[i], ploty[i]])
        pts_l_arr_f = np.array(pts_l_fit)
        pts_l_arr_f = pts_l_arr_f.reshape((-1,1,2))
        pts_r_arr_f = np.array(pts_r_fit)
        pts_r_arr_f = pts_r_arr_f.reshape((-1,1,2))
        cv2.polylines(out_img,np.int32([pts_l_arr_f]),False,(0,0,255),3)
        cv2.polylines(out_img,np.int32([pts_r_arr_f]),False,(0,0,255),3)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
         # Lane location at bottom of image
        left_lane_pos = pts[0][719][0]
        right_lane_pos= pts[0][720][0]
        
        lane_center = ( right_lane_pos + left_lane_pos ) / 2.
        lane_width = ( right_lane_pos - left_lane_pos )
        center_offset = 640 - lane_center
        center_offset_m = center_offset* 3.7 / 830
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, M_inv, (frame.shape[1], frame.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        avg_l_rad = sum(self.left_lane.last_n_radius)/len(self.left_lane.last_n_radius)
        avg_r_rad = sum(self.right_lane.last_n_radius)/len(self.right_lane.last_n_radius)
        l_rad_change = 100*( abs(self.left_lane.cur_radius - avg_l_rad))/avg_l_rad
        r_rad_change = 100*( abs(self.right_lane.cur_radius - avg_r_rad))/avg_r_rad

        if ( self.debug ):
            x = 0
            y = result.shape[0]/3 + 30
            text1 = "ROC_L:{:.2f}m, change of ({:.2f}%)".format(self.left_lane.cur_radius,l_rad_change)
            text2 = "ROC_R:{:.2f}m, change of ({:.2f}%)".format(self.right_lane.cur_radius,r_rad_change)
        else:
            x = 0
            y = 40
            text1 = "ROC_L:{:.2f}m".format(self.left_lane.cur_radius)
            text2 = "ROC_R:{:.2f}m".format(self.right_lane.cur_radius)

        text4 = "Vehicle offset:{:.2f}m".format(center_offset_m)
        
        cv2.putText(result, text1, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(result, text2, (int(x), int(y+40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(result, text4, (int(x), int(y+80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if ( self.debug ):
            y_size = int(result.shape[0]/3)
            x_size = int(result.shape[1]/3)
            warp_inset = cv2.resize(warped,(x_size,y_size))
            result[0:y_size,0:x_size] = warp_inset
            thres_inset = cv2.resize(thresholded_stack,(x_size,y_size))
            result[0:y_size,x_size:2*x_size] = thres_inset
            lane_inset = cv2.resize(out_img,(x_size,y_size))
            result[0:y_size,2*x_size:3*x_size] = lane_inset
            cv2.putText(result, "Perspective image", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(result, "Thresholded image", (x_size,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(result, "Lanes detected", (2*x_size,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return result
    
    def find_lanes_search_all(self, warped_binary, margin=75, find='ALL'):
        offset = 200
        x_start = offset
        x_end = warped_binary.shape[1] - offset
        x_mid = int (( x_start + x_end ) / 2 )
        self.count += 1
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(warped_binary.shape[0]/nwindows)

        if ( find == 'ALL' ):
            # Take a histogram of the bottom half of the image with some offset in x
            histogram = np.sum(warped_binary[int(warped_binary.shape[0]/2):,x_start:x_end], axis=0)
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint]) + offset
            rightx_base = np.argmax(histogram[midpoint:]) + x_mid
            # nonzero = warped_binary.nonzero()
            leftx_current = leftx_base
            rightx_current = rightx_base
        elif ( find == 'LEFT'):
            # Take a histogram of the bottom half of the image with some offset in x
            histogram = np.sum(warped_binary[int(warped_binary.shape[0]/2):,x_start:x_mid], axis=0)
            leftx_base = np.argmax(histogram) + offset
            # nonzero = warped_binary[:,:int(warped_binary.shape[1]/2)].nonzero()
            leftx_current = leftx_base
            rightx_current = None
        elif ( find == 'RIGHT'):
            # Take a histogram of the bottom half of the image with some offset in x
            histogram = np.sum(warped_binary[int(warped_binary.shape[0]/2):,x_mid:x_end], axis=0)
            rightx_base = np.argmax(histogram) + x_mid
            # nonzero = warped_binary[:,int(warped_binary.shape[1]/2):].nonzero()
            leftx_current = None
            rightx_current = rightx_base
        else:
            # Should not be here
            print ("Check the function call for find_lanes_search_all!")
            exit()

        # Plot the histogram. Uncomment the lines below: 
        # x = np.arange(histogram.shape[0])
        # plt.plot(x,histogram)
        # plt.show()

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(0,nwindows): 
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_binary.shape[0] - (window+1)*window_height
            win_y_high = warped_binary.shape[0] - window*window_height
            if ( find == 'ALL' or find == 'LEFT' ):
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if ( find == 'ALL' or find == 'RIGHT' ):
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        if ( len(left_lane_inds) > 0 ):
            left_lane_inds = np.concatenate(left_lane_inds)
            # Extract left and right line pixel positions
            self.left_lane.cur_x_cloud = nonzerox[left_lane_inds]
            self.left_lane.cur_y_cloud = nonzeroy[left_lane_inds]
            
            # self.left_lane.is_lane_good()
            self.left_lane.good_lane = True
            # Add the current cloud data to existing cloud data
            if ( len(self.left_lane.last_n_x_cloud) == self.left_lane.average_over_n_frames ):
                del self.left_lane.last_n_x_cloud[0]
                del self.left_lane.last_n_y_cloud[0]
            self.left_lane.last_n_x_cloud.append(self.left_lane.cur_x_cloud.tolist())
            self.left_lane.last_n_y_cloud.append(self.left_lane.cur_y_cloud.tolist())
            # Get single pixel values for each y point
            self.left_lane.get_single_points(False)
            # Fit a second order polynomial to each
            self.left_lane.get_poly_coeffs()
            # Calculate ROC (in m)
            self.left_lane.calculate_roc(max(self.left_lane.cur_y_points))
            if ( len(self.left_lane.last_n_radius) == self.left_lane.average_over_n_frames ):
                del self.left_lane.last_n_radius[0]
            self.left_lane.last_n_radius.append(self.left_lane.cur_radius)            
            # For targeted search next time            
            self.left_lane.prev_fit_coeff = self.left_lane.fit_coeff
            self.left_lane.skip_count = 0

        if ( len(right_lane_inds) > 0 ):
            right_lane_inds = np.concatenate(right_lane_inds)
            # Extract left and right line pixel positions
            self.right_lane.cur_x_cloud = nonzerox[right_lane_inds]
            self.right_lane.cur_y_cloud = nonzeroy[right_lane_inds]

            # self.right_lane.is_lane_good()
            self.right_lane.good_lane = True
            # Add the current cloud data to existing cloud data
            if ( len(self.right_lane.last_n_x_cloud) == self.right_lane.average_over_n_frames ):
                del self.right_lane.last_n_x_cloud[0]
                del self.right_lane.last_n_y_cloud[0]
            self.right_lane.last_n_x_cloud.append(self.right_lane.cur_x_cloud.tolist())
            self.right_lane.last_n_y_cloud.append(self.right_lane.cur_y_cloud.tolist())
            # Get single pixel values for each y point
            self.right_lane.get_single_points(False)
            # Fit a second order polynomial to each
            self.right_lane.get_poly_coeffs()
            # Calculate ROC (in m)
            self.right_lane.calculate_roc(max(self.right_lane.cur_y_points))
            if ( len(self.right_lane.last_n_radius) == self.right_lane.average_over_n_frames ):
                del self.right_lane.last_n_radius[0]            
            self.right_lane.last_n_radius.append(self.right_lane.cur_radius)
            # For targeted search next time
            self.right_lane.prev_fit_coeff = self.right_lane.fit_coeff
            self.right_lane.skip_count = 0 

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
        if ( find == 'ALL' or find == 'LEFT' ):
            left_fitx = self.left_lane.fit_coeff[0]*ploty**2 + \
                        self.left_lane.fit_coeff[1]*ploty + \
                        self.left_lane.fit_coeff[2]
        if ( find == 'ALL' or find == 'RIGHT' ):                
            right_fitx = self.right_lane.fit_coeff[0]*ploty**2 + \
                        self.right_lane.fit_coeff[1]*ploty + \
                        self.right_lane.fit_coeff[2]
        
        if ( find == 'ALL' ):
            return (left_fitx, right_fitx, ploty)
        elif ( find == 'LEFT' ):
            return (left_fitx, ploty)
        elif ( find == 'RIGHT' ):
            return (right_fitx, ploty)

    def find_lanes_search_targeted(self, warped_binary, margin=50, find=''):
        mid = int(warped_binary.shape[1]/2)
        nonzero = warped_binary.nonzero()

        nonzeroy = np.array(nonzero[0])
        if ( find == 'LEFT' ):
            nonzerox = np.array(nonzero[1])
        elif ( find == 'RIGHT' ):
            nonzerox = np.array(nonzero[1]) # + mid

        if ( find == 'LEFT' ):
            left_lane_inds = ((nonzerox > (self.left_lane.prev_fit_coeff[0]*(nonzeroy**2) + \
                                            self.left_lane.prev_fit_coeff[1]*nonzeroy + \
                                            self.left_lane.prev_fit_coeff[2] - margin)) \
                            & (nonzerox < (self.left_lane.prev_fit_coeff[0]*(nonzeroy**2) + \
                                            self.left_lane.prev_fit_coeff[1]*nonzeroy + \
                                            self.left_lane.prev_fit_coeff[2] + margin))) 
            self.left_lane.cur_x_cloud = nonzerox[left_lane_inds]
            self.left_lane.cur_y_cloud = nonzeroy[left_lane_inds]
            # Get single pixel values for each y point
            self.left_lane.get_single_points()
            # Fit a second order polynomial to each
            self.left_lane.get_poly_coeffs() 
            # Calculate ROC (in m)
            self.left_lane.calculate_roc(np.max(self.left_lane.cur_y_cloud))
            self.left_lane.is_lane_good()
            # Is this a good lane detection?
            if ( self.left_lane.good_lane ):
                # Add the current cloud data to existing cloud data
                if ( len(self.left_lane.last_n_x_cloud) == self.left_lane.average_over_n_frames ):
                    del self.left_lane.last_n_x_cloud[0]
                    del self.left_lane.last_n_y_cloud[0]                    
                self.left_lane.last_n_x_cloud.append(self.left_lane.cur_x_cloud.tolist())
                self.left_lane.last_n_y_cloud.append(self.left_lane.cur_y_cloud.tolist())
                # Get single pixel values for each y point
                self.left_lane.get_single_points(False)
                # Fit a second order polynomial to each
                self.left_lane.get_poly_coeffs()
                # Calculate ROC (in m)
                self.left_lane.calculate_roc(max(self.left_lane.cur_y_points))
                if ( len(self.left_lane.last_n_radius) == self.left_lane.average_over_n_frames ):
                    del self.left_lane.last_n_radius[0]
                self.left_lane.last_n_radius.append(self.left_lane.cur_radius)            
                # For targeted search next time            
                self.left_lane.prev_fit_coeff = self.left_lane.fit_coeff
                self.left_lane.skip_count = 0

                # Generate x and y values for plotting
                ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
                left_fitx = self.left_lane.fit_coeff[0]*ploty**2 + \
                            self.left_lane.fit_coeff[1]*ploty + \
                            self.left_lane.fit_coeff[2]
                
            else:
                # Get single pixel values for each y point
                self.left_lane.get_single_points(False)
                self.left_lane.skip_count += 1                
                # Use the last calculate fit coefficient
                ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
                left_fitx = self.left_lane.prev_fit_coeff[0]*ploty**2 + \
                            self.left_lane.prev_fit_coeff[1]*ploty + \
                            self.left_lane.prev_fit_coeff[2]
            return (left_fitx, ploty)

        if ( find == 'RIGHT' ):
            right_lane_inds = ((nonzerox > (self.right_lane.prev_fit_coeff[0]*(nonzeroy**2) + \
                                            self.right_lane.prev_fit_coeff[1]*nonzeroy + \
                                            self.right_lane.prev_fit_coeff[2] - margin)) \
                            & (nonzerox < (self.right_lane.prev_fit_coeff[0]*(nonzeroy**2) \
                                        + self.right_lane.prev_fit_coeff[1]*nonzeroy \
                                        + self.right_lane.prev_fit_coeff[2] + margin))) 
            self.right_lane.cur_x_cloud = nonzerox[right_lane_inds]
            self.right_lane.cur_y_cloud = nonzeroy[right_lane_inds]
            # Get single pixel values for each y point
            self.right_lane.get_single_points()
            # Fit a second order polynomial to each
            self.right_lane.get_poly_coeffs()
            if ( len ( self.right_lane.cur_x_points ) == 0 ):
                print("\n\n No points detected here!!\n\n")
            # Calculate ROC (in m)
            self.right_lane.calculate_roc(np.max(self.right_lane.cur_y_cloud))
            self.right_lane.is_lane_good()
            # Is this a good lane detection?
            if ( self.right_lane.good_lane ):
                # Add the current cloud data to existing cloud data
                if ( len(self.right_lane.last_n_x_cloud) == self.right_lane.average_over_n_frames ):
                    del self.right_lane.last_n_x_cloud[0]
                    del self.right_lane.last_n_y_cloud[0]                
                self.right_lane.last_n_x_cloud.append(self.right_lane.cur_x_cloud.tolist())
                self.right_lane.last_n_y_cloud.append(self.right_lane.cur_y_cloud.tolist())
                # Get single pixel values for each y point
                self.right_lane.get_single_points(False)
                # Fit a second order polynomial to each
                self.right_lane.get_poly_coeffs()
                # Calculate ROC (in m)
                self.right_lane.calculate_roc(max(self.right_lane.cur_y_points))
                if ( len(self.right_lane.last_n_radius) == self.right_lane.average_over_n_frames ):
                    del self.right_lane.last_n_radius[0]            
                self.right_lane.last_n_radius.append(self.right_lane.cur_radius)
                # For targeted search next time
                self.right_lane.prev_fit_coeff = self.right_lane.fit_coeff
                self.right_lane.skip_count = 0 

                # Generate x and y values for plotting
                ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
                right_fitx = self.right_lane.fit_coeff[0]*ploty**2 + \
                             self.right_lane.fit_coeff[1]*ploty + \
                             self.right_lane.fit_coeff[2]
            else:
                # Get single pixel values for each y point
                self.right_lane.get_single_points(False)
                self.right_lane.skip_count += 1            
                # Use the last calculate fit coefficient
                ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0] )
                right_fitx = self.right_lane.prev_fit_coeff[0]*ploty**2 + \
                            self.right_lane.prev_fit_coeff[1]*ploty + \
                            self.right_lane.prev_fit_coeff[2]

            return (right_fitx,ploty)