from PIL import Image
from pylab import *
from numpy import *

from PCV.tools import imtools, pca

# Get list of images and their size
imlist = imtools.get_imlist('fontimages/') # fontimages.zip is part of the book data set
im = array(Image.open(imlist[0])) # open one image to get the size 
m,n = im.shape[:2]

# Create matrix to store all flattened images
immatrix = array([array(Image.open(imname)).flatten() for imname in imlist],'f')

# Perform PCA
V,S,immean = pca.pca(immatrix)

# Show the images (mean and 7 first modes)
# This gives figure 1-8 (p15) in the book.
figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n))
for i in range(7):
    subplot(2,4,i+2)
    imshow(V[i].reshape(m,n))
show()
