import cv2
import numpy as np
import server
import sys

MIN_AREA=150 # the minimum area that the largest contour can be, otherwise findTarget returns None

def findTarget(image):
    '''Returns centre coordinates (x,y), dimensions (height, width),
    inclination angle, and the tweaked image (for manual/automatic 
    checking - not actually used by the robot itself).
    '''
    global MIN_AREA
    
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
    largest = []
    largest_area = 0 #store the area of largest here so we don't have to compute the area of the largest contour for every check
    for contour in contours:
        if cv2.contourArea(contour) > largest_area:
            largest = contour
            largest_area = cv2.contourArea(contour)

    if largest_area < MIN_AREA:
        return [0, 0, 0, 0, 0], image

    target_contour = largest
    
    #cv2.drawContours(blurred, [target_contour], 0, (0, 0, 255), -1)

    
    # We know our target is a rectangle. This means we can fit a bounding box
    # to it. We should make it an oriented bounding box (OBB) because if the
    # target is to the side it appears on an angle in our image.
    rect = cv2.minAreaRect(target_contour)
    obb_image = image
    obb = cv2.cv.BoxPoints(rect)
    obb = np.int0(obb)
    cv2.drawContours(obb_image, [obb], -1, (0,0,255), 3)
    
    # Now that we have an OBB we can get its vital stats to return to the
    # caller. Remember that these numbers need to be independent of the size
    # of the image (we can't return them in pixels). Scale everything relative
    # to the image - between [-1, 1]. So (1,1) would be the top right of the
    # image, (-1,-1) bottom left, and (0,0) dead centre.
    height, width, depth = image.shape
    #if nessacary fix the angle values and swap width and height
    if rect[1][0]>rect[1][1]:
        w=rect[1][0]/width
        h=rect[1][1]/width
        angle = rect[2]
    else:
        w=rect[1][1]/width
        h=rect[1][0]/width
        angle = 90+rect[2]
    x = 2*(rect[0][0]/width)-1
    y = 2*(rect[0][1]/height)-1
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
    
    return [x, y, w, h, angle], result_image
    
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
        cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS,0.2)
        cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, 0.9)

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
