import cv2
import numpy as np
import server
import sys
from collections import OrderedDict
import argparse
import ConfigParser
import urllib

def parseConfig():
    global configuration
    configuration = {}
    config = ConfigParser.ConfigParser()
    config.read("config.ini")
    if "target" in config.sections():
        target_items = config.items("target")
        for setting in target_items:
            print(setting)
            configuration[setting[0]] = eval(setting[1])
    

def findTarget(image):
    '''Returns centre coordinates (x,y), dimensions (configuration['height'], configuration['width']),
    inclination angle, and the tweaked image (for manual/automatic 
    checking - not actually used by the robot itself).
    '''
    global configuration
    
    # Use a Gaussian Blur to smooth out any ragged edges.
    #gaussian blur removed as lags framerates on BBB
    #blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # Convert from BGR colourspace to HSV. Makes thresholding easier.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Threshold the image to only get the green of the target. This gives us a
    # mask to apply to the original image (if we want).
    # Hue values are in degrees, but OpenCV only takes single byte arguments,
    # which means a maximum of 255. To get around this OpenCV takes hue values
    # in the range [0, 180]. This means 120 degrees (for example) maps to 60 in
    # OpenCV.
    lower = np.array(configuration['hsv_bounds'][0]) # get lower and upper values from numpy
    upper = np.array(configuration['hsv_bounds'][1])
    mask = cv2.inRange(hsv_image, lower, upper) # filter out all but values in the above range
    result = cv2.bitwise_and(image,image, mask=mask) # the image after thresholding

    # We can use the "opening" operation to remove noise from the mask.
    # Opening is an erosion then dilation.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    emask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=configuration['erode_dialate_iterations'])

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
        if area>target_area and area>(configuration['min_area']*image.shape[1]*image.shape[1]):# ifs are staggered for efficiency reasons
            rect = cv2.minAreaRect(contour)
            stats = get_data(rect, image)
            # if we have the right configuration['width'] to configuration['height'] ratio (within a cirtain tolerance) and the angle is within some threshold and our configuration['width'] to configuration['height'] ratio is within a cirtain range
            if -configuration['max_angle']<=stats['angle']<=configuration['max_angle'] and 2<stats['w']/stats['h']<configuration['ratio']*(1+configuration['target_tol']):
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

def parse_stream(stream):
    global img_bytes
    image = None
    got_frame = False
    while not got_frame:
        img_bytes+=stream.read(1024)
        a = img_bytes.find('\xff\xd8')
        b = img_bytes.find('\xff\xd9')
        if a!=-1 and b!=-1:
            print got_frame
            got_frame = True
            jpg = img_bytes[a:b+2]
            img_bytes= img_bytes[b+2:]
            image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
    return image

#get the vital stats out of an obb's rect
def get_data(rect, image):
    img_height, img_width, depth = image.shape
    if rect[1][0]>rect[1][1]:
        w=rect[1][0]/img_width
        h=rect[1][1]/img_width
        angle = rect[2]
        swapped = False # we have not swapped the configuration['width'] and configuration['height']
    else:
        w=rect[1][1]/img_width
        h=rect[1][0]/img_width
        angle = 90+rect[2]
    x = 2*(rect[0][0]/img_width)-1
    y = 2*(rect[0][1]/img_height)-1
    return OrderedDict([('x',x), ('y',y), ('w',w), ('h',h), ('angle',angle)])
    
if __name__ == "__main__":
    global img_bytes
    # Get the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--live", help="Display a live window", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--stream", help="Dump the processed image to stdout for streaming\nTypical command to pipe into:\navconv -f rawvideo -pix_fmt bgr24 -s 640x480 -r 30 -i - -an  -f mpegts udp://127.0.0.255:1234", action="store_true")
    group.add_argument("-v", "--verbose", help="Print output values on screen", action="store_true")
    parser.add_argument("-f", "--file", help="Load a test image from file")
    parser.add_argument("-w", "--webcam", help="Make the code receive frames from a webcam as opposed to an IP camera")

    args = parser.parse_args()
    
    parseConfig()
    
    img_bytes = ''
    stream = None
    processed_image = None
    to_send = OrderedDict([('x',0), ('y',0), ('w',0), ('h',0), ('angle',0)])
    
    if not args.file:
        if not args.webcam and not args.file:
            # Make a camera stream
            url = "http://root:pass@192.168.0.11/mjpg/video.mjpg?%s&%s&%s&%s" % ("brightness=" + str(configuration['brightness']), "contrast=" + str(configuration['contrast']), "&width" + str(configuration['width']), "&height" + str(configuration['height']))
            stream = urllib.urlopen(url)
        else:
            cap=cv2.VideoCapture(-1)
            cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, configuration['brightness'])
            cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, configuration['contrast'])
            cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, configuration['width'])
            cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, configuration['height'])

    while True:
        # Get an image from the camera
        if args.file:
            image = cv2.imread("img/target/" + args.file, -1)
        elif args.webcam:
            ret, image = cap.read()
        else:
            image = parse_stream(stream)
            
        if image is not None:
            to_send, processed_image = findTarget(image)

        if not to_send['w']:
            if args.verbose:
                print "No target found"
        else:
            if args.verbose:
                print "X:" + str(to_send['x']) + " Y:" + str(to_send['y']) + " Width:" + str(to_send['w']) + " Height:" + str(to_send['h']) + " Angle:" + str(to_send['angle'])
            #server.udp_send(to_send.values())
        if (args.live or args.file) and processed_image != None:
            cv2.imshow("Live Capture", processed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
