import unittest
import csv
import cv2
from ball import BallFinder

class TestBall(unittest.TestCase):


    def testFind(self):
        # TODO - read the files from the disk along with their parameters to pass to the tests
        
        bf = BallFinder()
        
        # At the moment we assume the csv file is in the format:
        # filename, colour, xbar, ybar, diam
        with open('img/ball.csv', 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                frame = cv2.imread('img/'+row[0])
                bf.setColour(row[1])
                bf.find(frame)
                self.assertAlmostEqual(bf.xbar, float(row[2]), delta=0.05)
                self.assertAlmostEqual(bf.ybar, float(row[3]), delta=0.05)
                self.assertAlmostEqual(bf.diam, float(row[4]), delta=0.05)
                


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()