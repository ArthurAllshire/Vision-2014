import cv2
import numpy as np
import server
import sys

RATIO = 23.5/4
TARGET_TOL = 0.1
MIN_AREA = 150
BRIGHTNESS = 0.2
CONTRAST = 0.9

def findTarget(image):
    '''Returns centre coordinates (x,y), dimensions (height, width),
    inclination angle, and the tweaked image (for manual/automatic 
    checking - not actually used by the robot itself).
    '''
    global RATIO, TARGET_TOL, MIN_AREA
    
    # Convert from BGR colourspace to HSV. Makes thresholding easier.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Threshold the image to only get the green of the target. This gives us a
    # mask to apply to the original image (if we want).
    # Hue values are in degrees, but OpenCV only takes single byte arguments,
    # which means a maximum of 255. To get around this OpenCV takes hue values
    # in the range [0, 180]. This means 120 degrees (for example) maps to 60 in
    # OpenCV.
    lower = np.array([85, 20, 150])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv_image, lower, upper)
    result = cv2.bitwise_and(image,image, mask=mask)

    # The thresholding will leave some ragged edges, and some rogue points in
    # the mask. Use a Gaussian Blur to smooth this out.
    blurred = cv2.GaussianBlur(result, (7, 7), 0)

    # OpenCV can find contours in the image - essentially closed loops of edges.
    # We are expecting the largest contour is the target.
    # First get all the contours:
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [0, 0, 0, 0, 0], image

    target_contour = []
    target_area = 0
    found_target = False
    stats = []
    rect = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        stats = get_data(rect, image)
        area = cv2.contourArea(contour)
        if stats[3] != 0: # make sure we do not get a divide by zero error as this can occur if stats[3] is zero
            if stats[2]/stats[3]*(1-TARGET_TOL) <= RATIO <= stats[2]/stats[3]*(1+TARGET_TOL) and area>largest_area and area>MIN_AREA:
                target_contour = contour
                found_target = True


    if not found_target:
        return [0, 0, 0, 0, 0], image
    
    #cv2.drawContours(blurred, [target_contour], 0, (0, 0, 255), -1)

    
    # We know our target is a rectangle. This means we can fit a bounding box
    # to it. We should make it an oriented bounding box (OBB) because if the
    # target is to the side it appears on an angle in our image.
    obb_image = image
    obb = cv2.cv.BoxPoints(rect)
    obb = np.int0(obb)
    cv2.drawContours(obb_image, [obb], -1, (0,0,255), 3)
    
    # Now that we have an OBB we can get its vital stats to return to the
    # caller. Remember that these numbers need to be independent of the size
    # of the image (we can't return them in pixels). Scale everything relative
    # to the image - between [-1, 1]. So (1,1) would be the top right of the
    # image, (-1,-1) bottom left, and (0,0) dead centre.
    """
    w = rect[1][0]/ width
    h = rect[1][1]/width
    angle = rect[2]
    """
    
    # We can return an altered image so that we can check that things are
    # working properly.
    ## result_image = <something with image and one of the intermediate steps -
    ##                 mask, blurred, contours, obb, etc>
    
    ####################
    # Dummy values to get it working
    #(x, y, w, h, angle) = (0, 0, 0, 0, 0)
    result_image = image
    ####################
    
    return stats, result_image

#get the vital stats out of a rectangle
def get_data(rect, image):
    img_height, img_width, depth = image.shape
    if rect[1][0]>rect[1][1]:
        w=rect[1][0]/img_width
        h=rect[1][1]/img_width
        angle = rect[2]
    else:
        print "i"
        w=rect[1][1]/img_width
        h=rect[1][0]/img_width
        angle = 90+rect[2]
    x = 2*(rect[0][0]/img_width)-1
    y = 2*(rect[0][1]/img_height)-1
    return [x, y, w, h, angle]
    
if __name__ == "__main__":
    # By default we operate in daemon mode and from live camera
    daemon = True
    from_file = False
    if len(sys.argv) != 1:
        # We have additional arguments - find the correct mode
        daemon = False
        if sys.argv[1] != 'live':
            # Load from the specified file
            from_file = True
            image = cv2.imread("img/target/" + sys.argv[1], -1)
    if not from_file:
        # Make a camera stream
        cap = cv2.VideoCapture(-1)
        cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, BRIGHTNESS)
        cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, CONTRAST)

    while True:
        # Get an image from the camera
        if not from_file:
            ret, image = cap.read()
        
        to_send, processed_image = findTarget(image)
        if not to_send[2]:
            if not daemon:
                print "No target found"
        else:
            if not daemon:
                print "X:" + str(to_send[0]) + " Y:" + str(to_send[1]) + " Width:" + str(to_send[2]) + " Height:" + str(to_send[3]) + " Angle:" + str(to_send[4])
                print cap.get(12)
            server.udp_send(to_send)
        if not daemon:
            cv2.imshow("Live Capture", processed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
