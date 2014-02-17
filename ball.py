import cv2
import cv2.cv as cv
import numpy as np

#Camera number
CAMERA = 1

#Kernel Dimensions
kernel = cv2.getStructuringElement (cv2.MORPH_ELLIPSE,(4, 4))

class BallFinder:
    def __init__(self, colour='r', width = 640, height = 480): # Constructor to get the video capture set up
        self._vc = cv2.VideoCapture(CAMERA)
        self._width = 1.0 * width # Force a float
        self._height = 1.0 * height
        
        #Set Video Capture properties
        self._vc.set(cv.CV_CAP_PROP_BRIGHTNESS, 0.5)
        self._vc.set(cv.CV_CAP_PROP_SATURATION, 0.5)  #0.125
        self._vc.set(cv.CV_CAP_PROP_CONTRAST, 0.5)   #0.01
        
        #Max. distance the camera can detect is 13 feet
        #DistanceThreshold is the distance threshold 
        #It is the maximum distance that the camera will detect
        self.DistanceThreshold = 13 #ft
        
        #distance formula from width of ball
        #distance = 1661.5*w**-1.075
        self._MaxWidth = (1661.5/self.DistanceThreshold)**(1/1.075)*width/640.0 #pixels

        #squareness ratio
        self.target_ratio = 0.6
        
        # Public member variables to store the last calculated values
        # Invalid values indicate ball not found
        self.xbar = 99.0 # xbar, ybar should be in the range [-1.0, 1.0]
        self.ybar = 99.0
        self.diam = 99.0 # diam should be in the range (0.0, 2.0]
        
        self._is_red = (colour[0] == 'r' or colour[0] == 'R')
        self._vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, self._width)
        self._vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self._height)

    def setColour(self, colour):
        self._is_red = (colour[0].lower() == 'r')

    def find(self):
        if not self._vc:
            # Try to reinitialise, but still return None
            self.__init__()
            return None
        # We have a video capture object so we can proceed
        retval, frame = self._vc.read()
        if not retval:
            self.xbar = 99.0 # xbar, ybar should be in the range [-1.0, 1.0]
            self.ybar = 99.0
            self.diam = 99.0
            return None
        # The capture was successful. Start processing
        hsv_image = cv2.cvtColor(frame, cv.CV_BGR2HSV)
        # Choose mask based on self._is_red
        if self._is_red:
            # Red alliance
            mask_neg = cv2.inRange(hsv_image, np.array((0, 50, 50)), np.array((10, 255, 255)))
            mask_pos = cv2.inRange(hsv_image, np.array((170, 50, 50)), np.array((180, 255, 255)))
            mask = mask_pos | mask_neg
        else:
            # Blue alliance
            mask = cv2.inRange(hsv_image, np.array((105, 50, 50)), np.array((130, 255, 255)))
        
        #Eroding and Dilating mask
        opened = cv2.erode(mask, kernel, iterations = 7)
        opened = cv2.dilate(opened, kernel, iterations = 7)
    
        contours, hierarchy = cv2.findContours(opened, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_NONE)
        largest_size = 0
        largest_index = 0
        
        ball_found = False
        if contours: 
            for index, contour in enumerate(contours):
                if cv2.contourArea(contour) > largest_size: 
                    
                    #get co-ordinates and dimensions
                    x,y,w,h = cv2.boundingRect(contours[largest_index])
                    
                    #ball squareness
                    ballratio = 1.0 * w/h
                    
                    if ballratio > self.target_ratio and ballratio < 1.0/self.target_ratio:
                        largest_size = cv2.contourArea(contour)
                        largest_index = index
                        
            if largest_size > 0:
                if w > self._MaxWidth:
                    moments = cv2.moments(contours[largest_index])
                    if moments['m00'] != 0:
                        self.xbar = 2.0*moments['m10']/moments['m00']/self._width - 1.0
                        self.ybar = 2.0*moments['m01']/moments['m00']/self._height - 1.0
                        x,y,w,h = cv2.boundingRect(contours[largest_index])
                        self.diam = (w + h)/ self._width
                        ball_found = True
                                    
        if not ball_found:
            # No ball found so set the member variables to invalid values
            self.xbar = 99.0
            self.ybar = 99.0
            self.diam = 99.0
        # Return the frame, the contours and largest image in case we
        # want to show them on the screen
        return (frame, contours, largest_index)
    
    def absolute(self):
        # Convert xbar, ybar and diam to absolute values for showing on screen
        return (int((self.xbar+1.0)*self._width/2.0),
            int((self.ybar+1.0)*self._height/2.0),
            int(self.diam*self._width/2.0))
    
if __name__ == "__main__":
    bf = BallFinder('r', 640, 480)
    cv2.namedWindow("preview")
    
    result = bf.find()
    while result != None:
        frame, contours, largest_index = result
        
        if bf.diam >= 0.0 and bf.diam <= 2.0:
            # Check that we found the ball
            
            # Draw what we think is the ball outline
            x, y, diam = bf.absolute() # Get screen plottable values
            
            cv2.circle(frame, (x, y), diam/2, (200,255,0), 3)
            # Draw the largest contour
            cv2.drawContours(frame, contours, largest_index, (0,255,0), 2) 
        
        
        cv2.imshow("preview", frame)
        
        key = cv2.waitKey (20)
        if key != -1: # Exit on any key
            break
        # Get the next frame, and loop forever
        result = bf.find()
