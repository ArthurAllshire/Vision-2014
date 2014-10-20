import csv
import cv2
from nose.tools import assert_almost_equal
from target import findTarget

def test_find():
    # At the moment we assume the csv file is in the format:
    # filename, x, y, w, h, angle
    cartesian_delta = 0.05 # Absolute (normalised screen size)
    dimension_delta = 0.20 # Relative (percentage) error
    angular_delta = 5.0 # Absolute (degrees)
    deltas = (cartesian_delta, dimension_delta, angular_delta)

    with open('img/target/target.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if row: # Ignore blank line at end of file
                frame = cv2.imread('img/target/'+row[0])
                result = findTarget(frame)
                desired = (row[1], row[2], row[3], row[4], row[5])
                # Create a separate test for each image
                yield find_target, row[0], result[:-1], desired, deltas # Don't send the return image
                
def find_target(filename, result, desired, deltas):
    message = filename + " - %s\nExpected: %s +/- %s\nReceived: %s"
    assert_almost_equal(result[0], float(desired[0]), delta=deltas[0], msg=message % ("x position", str(desired[0]), str(deltas[0]), str(result[0])))
    assert_almost_equal(result[1], float(desired[1]), delta=deltas[0], msg=message % ("y position", str(desired[1]), str(deltas[0]), str(result[1])))
    assert_almost_equal(result[2], float(desired[2]), delta=deltas[1]*float(desired[2]), msg=message % ("width", str(desired[2]), str(deltas[1]*float(desired[2])), str(result[2])))
    assert_almost_equal(result[3], float(desired[3]), delta=deltas[1]*float(desired[3]), msg=message % ("height", str(desired[3]), str(deltas[1]*float(desired[3])), str(result[3])))
    assert_almost_equal(result[4], float(desired[4]), delta=deltas[2], msg=message % ("angle", str(desired[4]), str(deltas[2]), str(result[4])))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    import nose
    nose.main()
