import csv
import cv2
from nose.tools import assert_almost_equal
from ball import BallFinder

def test_find():
    bf = BallFinder()
    
    # At the moment we assume the csv file is in the format:
    # filename, colour, xbar, ybar, diam
    with open('img/ball.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            frame = cv2.imread('img/'+row[0])
            bf.setColour(row[1])
            bf.find(frame)
            delta = 0.05
            # Create a separate test for each image
            yield find_ball, bf, row[0] + '(' + row[1] + ')', row[2], row[3], row[4], delta
                
def find_ball(bf, filename, xbar, ybar, diam, delta):
    message = filename + " - %s\nExpected: %s +/- "+str(delta)+"\nReceived: %s"
    assert_almost_equal(bf.xbar, float(xbar), delta=delta, msg=message % ("xbar", xbar, str(bf.xbar)))
    assert_almost_equal(bf.ybar, float(ybar), delta=delta, msg=message % ("ybar", ybar, str(bf.ybar)))
    assert_almost_equal(bf.diam, float(diam), delta=delta, msg=message % ("diam", diam, str(bf.diam)))
    
                


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    import nose
    nose.main()