from PIL import Image
from pylab import *
from scipy.ndimage import filters


"""
This is an example of unsharp masking using Gaussian blur.
"""

# load sample eye image from http://en.wikipedia.org/wiki/Unsharp_masking
im = array(Image.open("unsharpen.jpg").convert("L"), "f")

# create blurred version of the image
sigma = 3
blurred = filters.gaussian_filter(im,sigma)

# create unsharp version (no thresholding)
weight = 0.25
unsharp = im - 0.25*blurred


# plot the original and the unsharpened image
figure()
imshow(im)
gray()
title("original image")

figure()
imshow(unsharp)
gray()
title("unsharp mask with weight {}".format(weight))

show()
