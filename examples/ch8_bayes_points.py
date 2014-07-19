from pylab import *
from numpy import *
import pickle

from PCV.classifiers import bayes
from PCV.tools import imtools

"""
This is the simple 2D classification example in Section 8.2
using Bayes classifier.

If you have created the data files, it will reproduce the plot
in Figure 8-4.
"""


# load 2D points using Pickle
with open('../data/points_normal.pkl', 'r') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

# train Bayes classifier
bc = bayes.BayesClassifier()
bc.train([class_1,class_2], [1,-1])

# load test data using Pickle
with open('../data/points_normal_test.pkl', 'r') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

# test on the 10 first points
print bc.classify(class_1[:10])[0]

# define function for plotting
def classify(x, y, bc=bc):
    points = vstack((x,y))
    return bc.classify(points.T)[0]

# plot the classification boundary
imtools.plot_2D_boundary([-6,6,-6,6], [class_1,class_2], classify, [1,-1])
show()
