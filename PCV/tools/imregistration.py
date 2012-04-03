from PIL import Image
from xml.dom import minidom
from numpy import *
from pylab import *

from scipy import ndimage, linalg
from scipy.misc import imsave
import os

def read_points_from_xml(xmlFileName):
    """ Reads control points for face alignment. """

    xmldoc = minidom.parse(xmlFileName)
    facelist = xmldoc.getElementsByTagName('face')    
    faces = {}    
    for xmlFace in facelist:
        fileName = xmlFace.attributes['file'].value
        xf = int(xmlFace.attributes['xf'].value)
        yf = int(xmlFace.attributes['yf'].value)     
        xs = int(xmlFace.attributes['xs'].value)
        ys = int(xmlFace.attributes['ys'].value)
        xm = int(xmlFace.attributes['xm'].value)
        ym = int(xmlFace.attributes['ym'].value)
        faces[fileName] = array([xf, yf, xs, ys, xm, ym])
    return faces
    
    
def write_points_to_xml(faces, xmlFileName):
    xmldoc = minidom.Document()
    xmlFaces = xmldoc.createElement("faces")

    keys = faces.keys()
    for k in keys:
        xmlFace = xmldoc.createElement("face")
        xmlFace.setAttribute("file", k)
        xmlFace.setAttribute("xf", "%d" % faces[k][0])    
        xmlFace.setAttribute("yf", "%d" % faces[k][1])
        xmlFace.setAttribute("xs", "%d" % faces[k][2])    
        xmlFace.setAttribute("ys", "%d" % faces[k][3])
        xmlFace.setAttribute("xm", "%d" % faces[k][4])
        xmlFace.setAttribute("ym", "%d" % faces[k][5])
        xmlFaces.appendChild(xmlFace)

    xmldoc.appendChild(xmlFaces)

    fp = open(xmlFileName, "w")
    fp.write(xmldoc.toprettyxml(encoding='utf-8'))    
    fp.close()


def compute_rigid_transform(refpoints,points):
    """ Computes rotation, scale and translation for
        aligning points to refpoints. """
    
    A = array([    [points[0], -points[1], 1, 0],
                [points[1],  points[0], 0, 1],
                [points[2], -points[3], 1, 0],
                [points[3],  points[2], 0, 1],
                [points[4], -points[5], 1, 0],
                [points[5],  points[4], 0, 1]])

    y = array([    refpoints[0],
                refpoints[1],
                refpoints[2],
                refpoints[3],
                refpoints[4],
                refpoints[5]])
    
    # least sq solution to mimimize ||Ax - y||
    a,b,tx,ty = linalg.lstsq(A,y)[0]
    R = array([[a, -b], [b, a]]) # rotation matrix incl scale

    return R,tx,ty


def rigid_alignment(faces,path,plotflag=False):
    """    Align images rigidly and save as new images.
        path determines where the aligned images are saved
        set plotflag=True to plot the images. """
    
    # take the points in the first image as reference points
    refpoints = faces.values()[0]
    
    # warp each image using affine transform
    for face in faces:
        points = faces[face]
        
        R,tx,ty = compute_rigid_transform(refpoints, points)
        T = array([[R[1][1], R[1][0]], [R[0][1], R[0][0]]])    
        
        im = array(Image.open(os.path.join(path,face)))
        im2 = zeros(im.shape, 'uint8')
        
        # warp each color channel
        for i in range(len(im.shape)):
            im2[:,:,i] = ndimage.affine_transform(im[:,:,i],linalg.inv(T),offset=[-ty,-tx])
            
        if plotflag:
            imshow(im2)
            show()
            
        # crop away border and save aligned images
        h,w = im2.shape[:2]
        border = (w+h)/20
        
        # crop away border
        imsave(os.path.join(path, 'aligned/'+face),im2[border:h-border,border:w-border,:])

