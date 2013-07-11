from PIL import Image
from numpy import *

from scipy.ndimage import measurements,morphology

"""
This is the morphology counting objects example in Section 1.4.
"""

# load image and threshold to make sure it is binary
im = array(Image.open('../data/houses.png').convert('L'))
im = (im<128)

labels, nbr_objects = measurements.label(im)
print "Number of objects:", nbr_objects

# morphology - opening to separate objects better
im_open = morphology.binary_opening(im,ones((9,5)),iterations=2) 

labels_open, nbr_objects_open = measurements.label(im_open)
print "Number of objects:", nbr_objects_open