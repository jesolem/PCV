from PIL import Image
from pylab import *
from numpy import *
import os

from PCV.localdescriptors import dsift, sift
from PCV.tools import imtools

"""
This will process all hand gesture images with the dense SIFT descriptor.

Assumes you downloaded the hand images to ..data/hand_gesture.
The plot at the end generates one of the images of Figure 8-3.
"""

path = '../data/hand_gesture/train/'
imlist = []
for filename in os.listdir(path):
    if os.path.splitext(filename)[1] == '.ppm':
        imlist.append(path+filename)


# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename, featfile, 10, 5, resize=(50,50))


# show an image with features
l,d = sift.read_features_from_file(featfile)
im = array(Image.open(filename).resize((50,50)))
print im.shape

figure()
sift.plot_features(im, l, True)
show()
