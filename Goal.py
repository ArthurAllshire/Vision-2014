#Import libraries
import cv2
import cv2.cv as cv
import numpy as np
import math as m

class GoalFinder:
    def __init__(self, width = 640, height = 480): # Constructor to get the video capture set up
		#video Capture settings
        self.videoPort = 0 #change to index camera
        self._width = 1.0 * width # Force a float
        self._height = 1.0 * height

		#initialise video feed and make sure its good
		try: 
			self._vc = cv2.VideoCapture(self.videoPort)
		except:
			self._vc = None			
		
		#video resolution
        self._vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, self._width)
        self._vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self._height)

        #Set Video Capture properties
        self._vc.set(cv.CV_CAP_PROP_BRIGHTNESS, 0.2)
        self._vc.set(cv.CV_CAP_PROP_SATURATION, 0.5)
        self._vc.set(cv.CV_CAP_PROP_CONTRAST, 0.5)

        # Detected rectangles info
        self.rect_index = []
        self.rectangles = []
        self.defaultHvalue = 99.0
        self.rectHeight = self.defaultHvalue

        # Best position to fire from
        self.bestPosition = 5 # ft (position from goal)
        self.angle = self.defaultHvalue
        self.xpos = self.defaultHvalue
        self.ypos = self.defaultHvalue
        self.currentpos = self.defaultHvalue

        # Threshold to detect rectanlges
        self.threshold = 250
        self.maxval = 255

    def find(self):
        if not self._vc:
            # Try to reinitialise, but still return None
            self.__init__()
            return None
        # We have a video capture object so we can proceed
        retval, frame = self._vc.read()
        if not retval:
            return None

        # Do Goal Tracking Bit
        greyimage = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(greyimage)
        ret, thresh = cv2.threshold(equ, self.threshold, self.maxval, cv2.THRESH_BINARY)
        thresh2 = cv2.adaptiveThreshold(thresh, self.maxval,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,19,2)
        kernel = cv2.getStructuringElement (cv2.MORPH_RECT,(5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        blur = cv2.GaussianBlur(opened, (1,1), 0)
        edge = cv2.Canny(blur,0, self.maxval)
        dilopened = cv2.dilate(thresh, kernel, iterations = 2)

        contours, hierarchy = cv2.findContours(dilopened, cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_NONE)
        found_rectangles = []
        filter_height = []
        nothernstar = self.defaultHvalue
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1800:
                x,y,w,h = cv2.boundingRect(contours[index])
                found_rectangles.append([x,y,w,h])
                filter_height.append([h])
                filter_height = sorted(filter_height)
                if len(filter_height) != 0:
                   nothernstar = filter_height[len(filter_height)-1]
        self.rectangles = found_rectangles
        self.rectHeight = nothernstar
        self.rect_index = len(found_rectangles)

        #self.currentpos = (8*10**(-9)*(self.rectHeight**4))-(8*10**(-6)*(self.rectHeight**3))+(0.0032*(self.rectHeight**2))-(0.6061*self.rectHeight)+53.116
        self.angle = m.acos(self.bestPosition/self.currentpos) #in radians
        self.angle = (self.angle*m.pi/2)
        self.xpos = self.currentpos*m.cos(self.angle)
        self.ypos = self.currentpos*m.sin(self.angle)
        return 	frame #self.rectangles#, contours, index#, contours, largest_index

    def absolute(self):
        # Convert xbar, ybar and diam to absolute values for showing on screen
        return (int(self.angle),
            int(self.xpos),
            int(self.ypos))

if __name__ == "__main__":
    gf = GoalFinder( 640, 480)
    cv2.namedWindow("preview")
    while True:
        result = gf.find()
        frame = result
        for rect in gf.rectangles:
            x,y,w,h = rect
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 7)

        #frame, contours, largest_index = result\
        #cv2.drawContours(frame, contours, index, (0,255,0),
        cv2.imshow("preview", frame)
        print gf.absolute()
        key = cv2.waitKey (20)
        if key != -1:
			break# Exit on any keybreak
        # Get the next frame, and loop forever
