from PIL import Image
import numpy as np
import matplotlib.pyplot as pl

"""
This example shows how images are represented using pixels, color channels and data types.
"""


# read image to array
im = np.array(Image.open('../data/empire.jpg'))
print("Shape is: {0} of type {1}".format(im.shape, im.dtype))

# read grayscale version to float array
im = np.array(Image.open('../data/empire.jpg').convert('L'),'f')
print("Shape is: {0} of type {1}".format(im.shape, im.dtype))

# visualize the pixel value of a small region
col_1, col_2 = 190, 225
row_1, row_2 = 230, 265

# crop using array slicing
crop = im[col_1:col_2,row_1:row_2]
cols, rows = crop.shape

print("Created crop of shape: {0}".format(crop.shape))

# generate all the plots
pl.figure()
pl.imshow(im)
pl.gray()
pl.plot([row_1, row_2, row_2, row_1, row_1], [col_1, col_1, col_2, col_2, col_1], linewidth=2)
pl.axis('off')

pl.figure()
pl.imshow(crop)
pl.gray()
pl.axis('off')

pl.figure()
pl.imshow(crop)
pl.gray()
pl.plot(20*np.ones(cols), linewidth=2)
pl.axis('off')

pl.figure()
pl.plot(crop[20,:])
pl.ylabel("Graylevel value")

from mpl_toolkits.mplot3d import axes3d
fig = pl.figure()
ax = fig.gca(projection='3d')
# surface plot with transparency 0.5
X,Y = np.meshgrid(np.arange(cols),-np.arange(rows)) 
ax.plot_surface(X, Y, crop, alpha=0.5, cstride=2, rstride=2)

pl.show()