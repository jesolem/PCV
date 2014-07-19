from pylab import *
from numpy import *
from svmutil import *
import pickle

from PCV.tools import imtools

"""
This is the simple 2D classification example in Section 8.2
using a SVM classifier.

If you have created the data files, it will reproduce the plot
in Figure 8-5.
"""


# load 2D points using Pickle
with open('../data/points_normal.pkl', 'r') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

# convert to lists for libsvm
class_1 = map(list, class_1)
class_2 = map(list, class_2)
labels = list(labels)
samples = class_1+class_2 # concatenate the two lists

# create SVM
prob = svm_problem(labels, samples)
param = svm_parameter('-t 2')
# train SVM on data
m = svm_train(prob, param)

# how did the training do?
res = svm_predict(labels, samples, m)


# load test data using Pickle
with open('../data/points_normal_test.pkl', 'r') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

# convert to lists for libsvm
class_1 = map(list, class_1)
class_2 = map(list, class_2)


# define function for plotting
def predict(x, y, model=m):
    return array(svm_predict([0]*len(x), map(list, zip(x,y)), model)[0])

# plot the classification boundary
imtools.plot_2D_boundary([-6,6,-6,6], [array(class_1),array(class_2)], predict, [-1,1])
show()
