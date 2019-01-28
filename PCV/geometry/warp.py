import matplotlib.delaunay as md 
from scipy import ndimage
from pylab import *
from numpy import *

from PCV.geometry import homography
    

def image_in_image(im1,im2,tp):
    """ Put im1 in im2 with an affine transformation
        such that corners are as close to tp as possible.
        tp are homogeneous and counter-clockwise from top left. """ 
    
    # points to warp from
    m,n = im1.shape[:2]
    fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])
    
    # compute affine transform and apply
    H = homography.Haffine_from_points(tp,fp)
    im1_t = ndimage.affine_transform(im1,H[:2,:2],
                    (H[0,2],H[1,2]),im2.shape[:2])
    alpha = (im1_t > 0)
    
    return (1-alpha)*im2 + alpha*im1_t


def combine_images(im1,im2,alpha):
    """ Blend two images with weights as in alpha. """
    return (1-alpha)*im1 + alpha*im2    
    

def alpha_for_triangle(points,m,n):
    """ Creates alpha map of size (m,n) 
        for a triangle with corners defined by points
        (given in normalized homogeneous coordinates). """
    
    alpha = zeros((m,n))
    for i in range(min(points[0]),max(points[0])):
        for j in range(min(points[1]),max(points[1])):
            x = linalg.solve(points,[i,j,1])
            if min(x) > 0: #all coefficients positive
                alpha[i,j] = 1
    return alpha
    

def triangulate_points(x,y):
    """ Delaunay triangulation of 2D points. """
    
    centers,edges,tri,neighbors = md.delaunay(x,y)
    return tri


def plot_mesh(x,y,tri):
    """ Plot triangles. """ 
    
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]] # add first point to end
        plot(x[t_ext],y[t_ext],'r')


def pw_affine(fromim,toim,fp,tp,tri):
    """ Warp triangular patches from an image.
        fromim = image to warp 
        toim = destination image
        fp = from points in hom. coordinates
        tp = to points in hom.  coordinates
        tri = triangulation. """
                
    im = toim.copy()
    
    # check if image is grayscale or color
    is_color = len(fromim.shape) == 3
    
    # create image to warp to (needed if iterate colors)
    im_t = zeros(im.shape, 'uint8') 
    
    for t in tri:
        # compute affine transformation
        H = homography.Haffine_from_points(tp[:,t],fp[:,t])
        
        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:,:,col] = ndimage.affine_transform(
                    fromim[:,:,col],H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
        else:
            im_t = ndimage.affine_transform(
                    fromim,H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
        
        # alpha for triangle
        alpha = alpha_for_triangle(tp[:,t],im.shape[0],im.shape[1])
        
        # add triangle to image
        im[alpha>0] = im_t[alpha>0]
        
    return im
    
    
def panorama(H,fromim,toim,padding=2400,delta=2400):
    """ Create horizontal panorama by blending two images 
        using a homography H (preferably estimated using RANSAC).
        The result is an image with the same height as toim. 'padding' 
        specifies number of fill pixels and 'delta' additional translation. """ 
    
    # check if images are grayscale or color
    is_color = len(fromim.shape) == 3
    
    # homography transformation for geometric_transform()
    def transf(p):
        p2 = dot(H,[p[0],p[1],1])
        return (p2[0]/p2[2],p2[1]/p2[2])
    
    if H[1,2]<0: # fromim is to the right
        print('warp - right')
        # transform fromim
        if is_color:
            # pad the destination image with zeros to the right
            toim_t = hstack((toim,zeros((toim.shape[0],padding,3))))
            fromim_t = zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],
                                        transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the right
            toim_t = hstack((toim,zeros((toim.shape[0],padding))))
            fromim_t = ndimage.geometric_transform(fromim,transf,
                                    (toim.shape[0],toim.shape[1]+padding)) 
    else:
        print('warp - left')
        # add translation to compensate for padding to the left
        H_delta = array([[1,0,0],[0,1,-delta],[0,0,1]])
        H = dot(H,H_delta)
        # transform fromim
        if is_color:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0],padding,3)),toim))
            fromim_t = zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],
                                            transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0],padding)),toim))
            fromim_t = ndimage.geometric_transform(fromim,
                                    transf,(toim.shape[0],toim.shape[1]+padding))
    
    # blend and return (put fromim above toim)
    if is_color:
        # all non black pixels
        alpha = ((fromim_t[:,:,0] * fromim_t[:,:,1] * fromim_t[:,:,2] ) > 0)
        for col in range(3):
            toim_t[:,:,col] = fromim_t[:,:,col]*alpha + toim_t[:,:,col]*(1-alpha)
    else:
        alpha = (fromim_t > 0)
        toim_t = fromim_t*alpha + toim_t*(1-alpha)
    
    return toim_t

