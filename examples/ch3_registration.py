from PCV.tools import imregistration

"""
This is the face image registration example from Figure 3-6.
Make sure to create a folder 'aligned' under the jkfaces folder.
"""

# load the location of control points
xml_filename = '../data/jkfaces.xml'
points = imregistration.read_points_from_xml(xml_filename)

# register
imregistration.rigid_alignment(points,'../data/jkfaces/')