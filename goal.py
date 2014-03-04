#Import libraries
import cv2
import cv2.cv as cv
import math as m

#Camera number
CAMERA = 0

class GoalFinder:
    def __init__(self, width = 640, height = 480): # Constructor to get the video capture set up
        self._vc = cv2.VideoCapture(CAMERA)
        self._width = 1.0 * width # Force a float
        self._height = 1.0 * height
        self.center = 0.5* self._width

        #video resolution
        self._vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, self._width)
        self._vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self._height)

        #Set Video Capture properties
        self._vc.set(cv.CV_CAP_PROP_BRIGHTNESS, 0.2)
        self._vc.set(cv.CV_CAP_PROP_SATURATION, 0.5)
        self._vc.set(cv.CV_CAP_PROP_CONTRAST, 0.5)

        #Default values
        self.defaultHvalue = 99.0
        self.rectHeight = self.defaultHvalue

        # Best position to fire from
        self.rect_index = []
        self.gRange = self.defaultHvalue
        self.angle = self.defaultHvalue
        self.Hot = self.defaultHvalue
        self.currentPos = self.defaultHvalue
        self.currentWidth = self.defaultHvalue

        # Threshold to detect rectanlges
        self._threshold = 250
        self._maxval = 255
        self._minarea = 2.8125*self._width

        # Kernel for eroding/dilating
        self._kernel = cv2.getStructuringElement (cv2.MORPH_RECT,(5, 5))


    def find(self):
        if not self._vc:
            # Try to reinitialise, but still return None
            self.__init__()
            return None
        # We have a video capture object so we can proceed
        retval, frame = self._vc.read()
        if not retval:
            self.gRange = self.defaultHvalue
            self.angle = self.defaultHvalue
            self.Hot = self.defaultHvalue
            return None

        # Do Goal Tracking Bit
        #Convert to grey scale
        greyimage = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #Equalise to remove brightness
        equ = cv2.equalizeHist(greyimage)

        # Convert to binary
        ret, thresh = cv2.threshold(equ, self._threshold, self._maxval, cv2.THRESH_BINARY)
        thresh2 = cv2.adaptiveThreshold(thresh, self._maxval,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,19,2)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel)
        blur = cv2.GaussianBlur(opened, (1,1), 0)
        edge = cv2.Canny(blur,0, self._maxval)
        dilopened = cv2.dilate(thresh, self._kernel, iterations = 2)

        contours, hierarchy = cv2.findContours(dilopened, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_NONE)
        found_rectangles = []
        filter_height = []
        filter_x = []
        xvalue = []
        northernstar = self.defaultHvalue
        goal_found = False
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > self._minarea:
                x,y,w,h = cv2.boundingRect(contours[index])
                found_rectangles.append([x,y,w,h])
                filter_height.append([h])
                filter_x.append([x])
                if len(filter_height) != 0:
                    for index, w in enumerate(filter_height):
                        if w == max(filter_height):
                            xvalue = index - 1

                    filter_height = sorted(filter_height)
                    northernstar = filter_height[len(filter_height)-1]
                    angletoS = filter_x[0]
                    self.rectangles = found_rectangles	
                    if northernstar != 0:
                        self.rectHeight = northernstar[len(northernstar)-1]
                        self.rectWidth = angletoS[len(angletoS)-1]
                        self.rect_index = len(found_rectangles)

                        self.rectWidth = abs(self.rectWidth - self.center)
                        self.currentPos = 461.25*self.rectHeight**(-0.916) # in meters
                        self.currentWidth = 150.56*abs(self.rectWidth)**(-0.805)
                        self.angle = m.degrees(m.acos(-1 + (self.currentWidth**2/self.currentPos**2)))
                        self.gRange = self.currentPos
                        goal_found = True

        if not goal_found:
			self.gRange
			self.angle
			self.Hot

        return frame #self.rectangles#, contours, index#, contours, largest_index

    def absolute(self):
        # Convert xbar, ybar and diam to absolute values for showing on screen
        return (float(self.gRange),
            float(self.angle),
            int(self.Hot))

if __name__ == "__main__":
    gf = GoalFinder(640, 480)
    cv2.namedWindow("preview")
    result = gf.find()
    while result != None:
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
        result = gf.find()