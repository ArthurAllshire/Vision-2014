import numpy as np
import cv2
import urllib

class Driver():
    def __init__(self, axis_ip):
        
        try:
            url = "http://" + axis_ip + "/mjpg/video.mjpg" # construct the url of the stream
            self.stream = urllib.urlopen(url)
        except IOError:
            print "Error: connection to camera failed"
        
        self.clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(16,16)) # set the histogram equalization
        self.img_bytes = '' #needs to be initialized for parse_stream
    
    def parse_stream(self):
        got_frame = False
        while not got_frame: # until we have read and constructed the frame
            self.img_bytes+=self.stream.read(1024) # read the next 1024 bytes from the camera
            a = self.img_bytes.find('\xff\xd8') # marks the beggining of a jpg frame
            b = self.img_bytes.find('\xff\xd9') # marks the end of a jpg frame
            if a!=-1 and b!=-1: # if we have read the whole frame
                got_frame = True
                jpg = self.img_bytes[a:b+2] # extract the bytes of the jpeg
                self.img_bytes= self.img_bytes[b+2:] # clear the jpeg buffer
                self.image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR) # turn the image from a string into a jpg image
        
    def display_loop(self):
        while True:
            self.parse_stream()
            
            result = self.image
            
            result[:, :, 0] = self.clahe.apply(self.image[:, :, 0]) 
            
            if self.image != None:
                cv2.imshow("Equalized", self.image)
                cv2.imshow("Normal", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
                    
driver = Driver("192.168.0.11")

driver.display_loop()
