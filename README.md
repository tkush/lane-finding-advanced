# Advance Lane Finding
The goals of this code are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle
  position.

### Camera Calibration
Camera calibration is achieved by preparing "object points", which are the (x, y, z) coordinates of the
chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such
that the object points are the same for each calibration image. Thus, objp is just a replicated array of
coordinates, and objpoints will be appended with a copy of it every time I successfully detect all
chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the
corners in the image plane with each successful chessboard detection.
I then used the output objpoints and imgpoints to compute the camera calibration and distortion
coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test
image using the `cv2.undistort()` function and obtained this result:

![Distortion](/images/distortion.png)

The distortion matrix and coefficients calculated through these calibration images is then stored to disk so
the calibration does not have to be done each time for each image.

### Pipeline (single images)
Once the distortion coefficients are saved to disk, the `cv2.undistort()` function is applied to each frame
image to get an undistorted image. This is illustrated below:

![Undistorted image](/images/undistorted.png)

The difference between the two images is most noticeable around the corners, where the effects of camera
distortion are maximum.

The perspective of the camera image is transformed to get a “top-down” view of the lane. This is done in
several steps:
* Identify source points in the original image (that capture the lane area)
* Identify a rectangle in the transformed image to which the points from above will be mapped.
* Use the function `cv2.getPerspectiveTranform` to identify a mapping that transforms the
  source points to the destination points
* Use the function `cv2.warpPerspective` to warp the original image as shown below

![Undistorted image Perspective transformed](/images/perspective.png)

To ensure a good perspective transform (parallel lines in the original image remain parallel in the
transformed image), the source points were chosen for an image frame where the lanes are straight ahead.
After transforming this, a visual check is made to ensure that the transformed lines are indeed straight. This
mapping is then used for the entirety of the project video. The source and destination points chosen were
hardcoded and are listed below:
`src_points = np.float32([[220,700],[590,450],[690,450],[1090,700]])`
`dst_points = np.float32([[300,720],[300,0],[800,0],[800,720]])`

The perspective transform is done within the `frame_lane.py` file in the method `process_frame`.
**Key insight**: To better threshold the image, it was found that taking a perspective transform before
thresholding the image produced better results. This is because the transformed image contains less
irrelevant content and so is easier to threshold.

The goal of thresholding the image is to clearly identify the lane markings. There are two kinds of markings in
the project video – yellow and white lane markings. To investigate what colorspace identifies these markings
the best, individual channels in the following colorspaces were investigated:

![HSV](/images/hsv.png)

![HLS](/images/hls.png)

![YCrCb](/images/ycrcb.png)

![Lab](/images/lab.png)

The white lane markings are identified very well in the L channel within the HLS colorspace for a variety of
lighting/road conditions. Also, this channel is not very sensitive to shadows.
The yellow lane markings are identified very well in the Cb channel within the YCrCb colorspace and within
the b channel of the Lab colorspace. The Cb channel is chosen here.
The input image (perspective transformed) is first converted to HLS and YCrCb colorspace. The “L” channel
and “Cb” channels are isolated and a threshold range is applied to them as follows:
`l_thres = Image_Processing.color_thres(hls[:,:,1],[190,255])`
`cb_thres = Image_Processing.color_thres(ycrcb[:,:,2],[65,110])`
The function `Image_Processing.color_thres` converts the channel into a binary image based on the
above pixel values. This is achieved in the file `image_processing.py` within the method `color_thres`.
An example of a thresholded binary image is shown below.
![Perspective transformed Thresholded binary image](/images/binary.png)

Two methods are used to find lane lines from within the thresholded image:
1. Full search of the entire image: 
This method uses the sliding window approach to find the points that qualify as the lane lines. This approach is implemented in the method `find_lanes_search_all` in `frame_lane.py`.
This method is called to find the lanes when:
* Lanes were not reliably detected in the previous frame.
* Lanes are to be detected for the first time in the video
2. Targeted search for lanes: 
This method uses the lane found from the previous frame and searches
around that area within a margin to find the new lane. This method is used when a lane has already
been detected reliably. This approach is implemented in the method
`find_lanes_search_targeted` in `frame_lane.py`
What is a reliable lane detection? Once a lane has been found for the first time, it’s radius of curvature is kept
in memory and any subsequent lanes found are compared to the previous lanes. If the radius of curvature is
within some specified tolerance, then the new lane is termed as a reliable detection. To ensure reliable
detection always, the following logic is used:
**Step 1**: Find lanes through the sliding window approach for the first image in the video stream. Store the fit
coefficients, x and y pixels identified in lanes and the calculated lane curvatures in the Frame class
**Step 2**: Find lanes in the next image frame through a targeted search within a margin of the previous lanes
found. If this is a reliable detection, store the lane data into the Frame class. If it is not a reliable detection,
use the lanes found from the last frame.
**Step 3**: If a reliable lane is not found more than 5 times, force a new detection through the sliding window
approach. If this method finds a reliable lane, store the lane data for the future.
**Step 4**: Repeat from Step 2.
When (x,y) pixels are identified as lane candidates, a second order polynomial is fit through these points as y
= f(x) and this is used to draw the lane lines. The lanes found are shown below:

![Lanes detected](/images/lanesdetected.png)

Once the lane polynomials are identified for both left and right lanes, the radius of curvature is easily
calculated using the formula presented in the course material. The calculation is done in the method
`calculate_roc` in `frame_lane.py`. The radius of curvature is evaluated at the bottom of the image or for
the maximum Y position.
The lane width is calculated by averaging the location of the left and right lanes. The vehicle offset is
calculated by subtracting the center of the lane from the center of the image. This is converted to meters by
multiplying with the m/pixel measurement taken by investigating the output picture.

An inverse mapping is calculated to map the lane lines back on the original image. This is done inside
frame_lane.py at the end of the function process_frame. Below is an example of the lane lines mapped
on to the original image.

![Lanes found](/images/lanesfound.png)

The resulting video can be found [here](https://youtu.be/hC0hlfRuxT4).

### Discussion
1. Debugging mode
To aid in debugging the project I have included code to produce an output video that contains a lot more
information and is helpful in debugging. Here is a snapshot of the debugging video:

![Debug frame](/images/debug.png)

In this frame, the following information can be found:
* Perspective transformed image (top left)
* Thresholded perspective image (top middle) with left (blue) and right (red) lane point clouds
  detected
* Lanes identified (top right)
* Lanes plotted on the original image (bottom)
* Radius of curvature (ROC) for both left and right lanes in m
* % difference in the ROC calculation from the last good measurement
* Vehicle offset in m
The debug video can be found [here](https://youtu.be/Ctr4QL98HcY)

#### Areas of improvement and possible failure
The current approach is tailored to the project video and is not very generalized. The points for perspective
transformation are hardcoded and should ideally be computer algorithmically for each frame (or at least each
video).
Further, a more robust method to threshold the images may be required since the thresholding has been fine
tuned for the project video and is again not generalized for all road/lighting conditions.
I do not expect this pipeline to work robustly on video with significantly different lane markings, lighting
conditions etc.
