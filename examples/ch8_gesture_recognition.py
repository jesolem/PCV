from PIL import Image
from numpy import *
from pylab import *

from svmutil import *
import os

from PCV.classifiers import knn, bayes
from PCV.tools import imtools
from PCV.localdescriptors import dsift, sift


"""
This script collects the three classifiers applied to the hand gesture
recognition test.

If you have created the data files with the dsift features, this will
reproduce all confucion matrices in Chapter 8.

Use the if statements at the bottom to turn on/off different combinations.
"""


def read_gesture_features_labels(path):
    # create list of all files ending in .dsift
    featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]

    # read the features
    features = []
    for featfile in featlist:
        l,d = sift.read_features_from_file(featfile)
        features.append(d.flatten())
    features = array(features)

    # create labels
    labels = [featfile.split('/')[-1][0] for featfile in featlist]

    return features,array(labels)


def convert_labels(labels,transl):
    """ Convert between strings and numbers. """
    return [transl[l] for l in labels]


def print_confusion(res,labels,classnames):

    n = len(classnames)

    # confusion matrix
    class_ind = dict([(classnames[i],i) for i in range(n)])

    confuse = zeros((n,n))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]],class_ind[test_labels[i]]] += 1

    print 'Confusion matrix for'
    print classnames
    print confuse


# read training data
####################
features,labels = read_gesture_features_labels('../data/hand_gesture/train/')
print 'training data is:', features.shape, len(labels)

# read test data
####################
test_features,test_labels = read_gesture_features_labels('../data/hand_gesture/test/')
print 'test data is:', test_features.shape, len(test_labels)

classnames = unique(labels)
nbr_classes = len(classnames)


if False:
    # reduce dimensions with PCA
    from PCV.tools import pca

    V,S,m = pca.pca(features)

    # keep most important dimensions
    V = V[:50]
    features = array([dot(V,f-m) for f in features])
    test_features = array([dot(V,f-m) for f in test_features])


if True:
    # test kNN
    k = 1
    knn_classifier = knn.KnnClassifier(labels,features)
    res = array([knn_classifier.classify(test_features[i],k) for i in range(len(test_labels))]) # TODO kan goras battre


if False:
    # test Bayes
    bc = bayes.BayesClassifier()
    blist = [features[where(labels==c)[0]] for c in classnames]

    bc.train(blist,classnames)
    res = bc.classify(test_features)[0]


if False:
    # test SVM
    # convert to lists for libsvm
    features = map(list,features)
    test_features = map(list,test_features)

    # create conversion function for the labels
    transl = {}
    for i,c in enumerate(classnames):
        transl[c],transl[i] = i,c

    # create SVM
    prob = svm_problem(convert_labels(labels,transl),features)
    param = svm_parameter('-t 0')

    # train SVM on data
    m = svm_train(prob,param)

    # how did the training do?
    res = svm_predict(convert_labels(labels,transl),features,m)

    # test the SVM
    res = svm_predict(convert_labels(test_labels,transl),test_features,m)[0]
    res = convert_labels(res,transl)


# accuracy
acc = sum(1.0*(res==test_labels)) / len(test_labels)
print 'Accuracy:', acc

print_confusion(res,test_labels,classnames)
