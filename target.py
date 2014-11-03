import cv2
import numpy as np
import server
import sys
from collections import OrderedDict
import argparse

RATIO = 23.5/4 # width/height
TARGET_TOL = 0.6 # the tolerance in the target width to height ratio
MIN_AREA = 300.0/(640**2) # the minimum area of any contours that will be kept in the image
BRIGHTNESS = 0.2 # the brightness camera property
CONTRAST = 0.9 # the contrast camera property
WIDTH, HEIGHT = 640, 480 # making this smaller will improve speed
TARGET_CUTOFF = 45 # we will not find the target if it is at an angle greater than this
ERODE_DIALATE_ITERATIONS = 2 # iteerations of erosion adn dialation
HSV_BOUNDS = ([50, 20, 20], [80, 255, 255]) # the lower and upper bounds of the values that will be kept after thresholding
MAX_ANGLE = 45 # the maximum (and its negative the minimum) angle that a contour can be at to be considered as elegable to be a target

def findTarget(image):
    '''Returns centre coordinates (x,y), dimensions (height, width),
    inclination angle, and the tweaked image (for manual/automatic 
    checking - not actually used by the robot itself).
    '''
    global RATIO, TARGET_TOL, MIN_AREA, TARGET_CUTOFF, ERODE_DIALATE_ITERATIONS, HSV_BOUNDS
    
    # Use a Gaussian Blur to smooth out any ragged edges.
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # Convert from BGR colourspace to HSV. Makes thresholding easier.
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Threshold the image to only get the green of the target. This gives us a
    # mask to apply to the original image (if we want).
    # Hue values are in degrees, but OpenCV only takes single byte arguments,
    # which means a maximum of 255. To get around this OpenCV takes hue values
    # in the range [0, 180]. This means 120 degrees (for example) maps to 60 in
    # OpenCV.
    lower = np.array(HSV_BOUNDS[0]) # get lower and upper values from numpy
    upper = np.array(HSV_BOUNDS[1])
    mask = cv2.inRange(hsv_image, lower, upper) # filter out all but values in the above range
    result = cv2.bitwise_and(image,image, mask=mask) # the image after thresholding

    # We can use the "opening" operation to remove noise from the mask.
    # Opening is an erosion then dilation.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    emask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=ERODE_DIALATE_ITERATIONS)

    erd_dlt_image = cv2.bitwise_and(image,image, mask=mask) #image after errode and dialate

    # OpenCV can find contours in the image - essentially closed loops of edges.
    # We are expecting the largest contour is the target.
    # First get all the contours:
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return OrderedDict([('x',0), ('y',0), ('w',0), ('h',0), ('angle',0)]), image

    target_contour = None
    target_rect = None
    target_area = 0
    found_target = False
    stats = []
    rect = []
    #Then sanity checks to find the largest contour that has the correct dimentions, area and side ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        if area>target_area and area>(MIN_AREA*image.shape[1]*image.shape[1]):# ifs are staggered for efficiency reasons
            rect = cv2.minAreaRect(contour)
            stats = get_data(rect, image)
            # if we have the right width to height ratio (within a cirtain tolerance) and the angle is within some threshold and our width to height ratio is within a cirtain range
            if -MAX_ANGLE<=stats['angle']<=MAX_ANGLE and 2<stats['w']/stats['h']<RATIO*(1+TARGET_TOL):
                #print "found one"
                target_contour = contour
                target_area = area
                found_target = True
                target_rect = rect

    # Plot all the contours to see how much noise we had, and where the contours
    # were found.
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

    if not found_target:
        return OrderedDict([('x',0), ('y',0), ('w',0), ('h',0), ('angle',0)]), image

    # We know our target is a rectangle. This means we can fit a bounding box
    # to it. We should make it an oriented bounding box (OBB) because if the
    # target is to the side it appears on an angle in our image.
    obb_image = image
    obb = cv2.cv.BoxPoints(target_rect)
    obb = np.int0(obb)
    cv2.drawContours(obb_image, [obb], -1, (0,0,255), 3)
    
    result_image = image
    
    return stats, result_image

#get the vital stats out of an obb's rect
def get_data(rect, image):
    global RATIO
    img_height, img_width, depth = image.shape
    if rect[1][0]>rect[1][1]:
        w=rect[1][0]/img_width
        h=rect[1][1]/img_width
        angle = rect[2]
        swapped = False # we have not swapped the width and height
    else:
        w=rect[1][1]/img_width
        h=rect[1][0]/img_width
        angle = 90+rect[2]
    x = 2*(rect[0][0]/img_width)-1
    y = 2*(rect[0][1]/img_height)-1
    return OrderedDict([('x',x), ('y',y), ('w',w), ('h',h), ('angle',angle)])
    
if __name__ == "__main__":
    # Get the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--live", help="Display a live window", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--stream", help="Dump the processed image to stdout for streaming\nTypical command to pipe into:\navconv -f rawvideo -pix_fmt bgr24 -s 640x480 -r 30 -i - -an  -f mpegts udp://127.0.0.255:1234", action="store_true")
    group.add_argument("-v", "--verbose", help="Print output values on screen", action="store_true")
    parser.add_argument("-f", "--file", help="Load a test image from file")

    args = parser.parse_args()
    
    if not args.file:
        # Make a camera stream
        cap = cv2.VideoCapture(-1)
        cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, BRIGHTNESS)
        cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, CONTRAST)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, HEIGHT)


    while True:
        # Get an image from the camera
        if args.file:
            image = cv2.imread("img/target/" + args.file, -1)
        else:
            ret, image = cap.read()
        
        to_send, processed_image = findTarget(image)
        if not to_send['w']:
            if args.verbose:
                print "No target found"
        else:
            if args.verbose:
                print "X:" + str(to_send['x']) + " Y:" + str(to_send['y']) + " Width:" + str(to_send['w']) + " Height:" + str(to_send['h']) + " Angle:" + str(to_send['angle'])
            server.udp_send(to_send.values())
        if args.live or args.file:
            cv2.imshow("Live Capture", processed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if args.stream:
            sys.stdout.write( processed_image.tostring() )

