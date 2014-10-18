import cv2
import numpy as np

def findTarget(image):
    '''Returns centre coordinates (x,y), dimensions (height, width),
    inclination angle, and the tweaked image (for manual/automatic 
    checking - not actually used by the robot itself).
    '''
    
    # Convert from BGR colourspace to HSV. Makes thresholding easier.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Threshold the image to only get the green of the target. This gives us a
    # mask to apply to the original image (if we want).
    # Hue values are in degrees, but OpenCV only takes single byte arguments,
    # which means a maximum of 255. To get around this OpenCV takes hue values
    # in the range [0, 180]. This means 120 degrees (for example) maps to 60 in
    # OpenCV.
    lower = np.array([88, 30, 150])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv_image, lower, upper)
    result = cv2.bitwise_and(image,image, mask=mask)

    # The thresholding will leave some ragged edges, and some rogue points in
    # the mask. Use a Gaussian Blur to smooth this out.
    blurred = cv2.GaussianBlur(result, (7, 7), 0)

    # OpenCV can find contours in the image - essentially closed loops of edges.
    # We are expecting the largest contour is the target.
    # First get all the contours:
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest = []
    largest_area = 0 #store the area of largest here so we don't have to compute the area of the largest contour for every check
    for contour in contours:
        if cv2.contourArea(contour) > largest_area:
            largest = contour
            largest_area = cv2.contourArea(contour)
    target_contour = [largest]
    cv2.drawContours(blurred, target_contour, 0, (0, 0, 255), -1)
    
    # Now find the largest and do some sanity checking to make sure we have
    # actually got a target in view.
    ##largest = ???
    ## <insert sanity checking here>
    
    # We know our target is a rectangle. This means we can fit a bounding box
    # to it. We should make it an oriented bounding box (OBB) because if the
    # target is to the side it appears on an angle in our image.
    ##obb = ???
    
    # Now that we have an OBB we can get its vital stats to return to the
    # caller. Remember that these numbers need to be independent of the size
    # of the image (we can't return them in pixels). Scale everything relative
    # to the image - between [-1, 1]. So (1,1) would be the top right of the
    # image, (-1,-1) bottom left, and (0,0) dead centre.
    ## x = ???
    ## y = ???
    ## w = ???
    ## h = ???
    ## angle = ???
    
    # We can return an altered image so that we can check that things are
    # working properly.
    ## result_image = <something with image and one of the intermediate steps -
    ##                 mask, blurred, contours, obb, etc>
    
    ####################
    # Dummy values to get it working
    (x, y, w, h, angle) = (0, 0, 0, 0, 0)
    result_image = blurred#image
    ####################
    
    return x, y, w, h, angle, result_image
    
if __name__ == "__main__":
    # Load an image. This could be a test image from a file,
    # or a frame from the video stream
    # Store it in a variable called 'image'
    
    image = cv2.imread("img/target/45degrees/D3.jpg", -1)
    """
    ################
    # Dummy image to get it going
    import numpy as np
    image = np.zeros((480, 640, 3), np.uint8)
    image[:] = (0, 180, 0) # BGR
    ################
     """
    
    x, y, w, h, angle, processed_image = findTarget(image)
    print processed_image
    cv2.namedWindow("preview")
    cv2.imshow("preview", processed_image)
    
    # Wait for a key, or the preview image just disappears...
    key = cv2.waitKey(0)
