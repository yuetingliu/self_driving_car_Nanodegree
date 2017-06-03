# **Finding Lane Lines on the Road** 

## Writeup 

### Reflection

### 1. The pipeline

My pipeline consists of 5 steps. Firstly, I converted the images to grayscale. Secondly, I applied Gaussian smoothing and Canny to find the edges. Then, I masked the image with a trapezoidal area, followed by applying Hough transforming of the masked area to locate the lane lines. Finally, I drew the lane lines and overlayed the image with the original image.  

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by adding three extra steps. Step 1 is to split all lines (got from Hough transform) into left lane and right lane by their slopes. Step 2 is to filter out noise/outliers by removing slopes that deviate from the median slope more than 2 std. Step 3 is to average the residual slopes and intercepts and to draw two continuous straight lane lines


### 2. Some shortcomings with the current pipeline

The first shortcoming is about the filters. I used slope and standard deviation as filters to remove noise. However this is a bit 'optimized' (overfitting, to be honest). If the noise line aligns with the lane line, then it cannot be filtered out by neither slope nor std. In the challenge video of this project, the shadow of the trees rather deviate from the lane a lot, therefore they are easy, but I can imagine in real situation, it will be difficult. 

Another shortcoming is that it is not always appropriate to draw straight lines. For instance, in a corner, the lane lines are typically curved somewhat. Besides, Hough transforming may have a hard time finding any line signal. For the challenge video, I have to use a try-except clause in draw_line() function to prevent unnecessary break of processing.  


### 3. Possible improvements to the current pipeline

A possible improvement would be to better technique to remove noise like shadows.uuuse

Another potential improvement could be to conditionally draw curved lines for situations like corners.
