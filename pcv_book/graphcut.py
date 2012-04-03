from pylab import *
from numpy import *

from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow

import bayes

""" 
Graph Cut image segmentation using max-flow/min-cut. 
"""

def build_bayes_graph(im,labels,sigma=1e2,kappa=1):
    """    Build a graph from 4-neighborhood of pixels. 
        Foreground and background is determined from
        labels (1 for foreground, -1 for background, 0 otherwise) 
        and is modeled with naive Bayes classifiers."""
    
    m,n = im.shape[:2]
    
    # RGB vector version (one pixel per row)
    vim = im.reshape((-1,3))
    
    # RGB for foreground and background
    foreground = im[labels==1].reshape((-1,3))
    background = im[labels==-1].reshape((-1,3))    
    train_data = [foreground,background]
    
    # train naive Bayes classifier
    bc = bayes.BayesClassifier()
    bc.train(train_data)

    # get probabilities for all pixels
    bc_lables,prob = bc.classify(vim)
    prob_fg = prob[0]
    prob_bg = prob[1]
    
    # create graph with m*n+2 nodes
    gr = digraph()
    gr.add_nodes(range(m*n+2))
    
    source = m*n # second to last is source
    sink = m*n+1 # last node is sink

    # normalize
    for i in range(vim.shape[0]):
        vim[i] = vim[i] / (linalg.norm(vim[i]) + 1e-9)
    
    # go through all nodes and add edges
    for i in range(m*n):
        # add edge from source
        gr.add_edge((source,i), wt=(prob_fg[i]/(prob_fg[i]+prob_bg[i])))
        
        # add edge to sink
        gr.add_edge((i,sink), wt=(prob_bg[i]/(prob_fg[i]+prob_bg[i])))
        
        # add edges to neighbors
        if i%n != 0: # left exists
            edge_wt = kappa*exp(-1.0*sum((vim[i]-vim[i-1])**2)/sigma)
            gr.add_edge((i,i-1), wt=edge_wt)
        if (i+1)%n != 0: # right exists
            edge_wt = kappa*exp(-1.0*sum((vim[i]-vim[i+1])**2)/sigma)
            gr.add_edge((i,i+1), wt=edge_wt)
        if i//n != 0: # up exists
            edge_wt = kappa*exp(-1.0*sum((vim[i]-vim[i-n])**2)/sigma)
            gr.add_edge((i,i-n), wt=edge_wt)
        if i//n != m-1: # down exists
            edge_wt = kappa*exp(-1.0*sum((vim[i]-vim[i+n])**2)/sigma)
            gr.add_edge((i,i+n), wt=edge_wt)
        
    return gr    
        
    
def cut_graph(gr,imsize):
    """    Solve max flow of graph gr and return binary 
        labels of the resulting segmentation."""
    
    m,n = imsize
    source = m*n # second to last is source
    sink = m*n+1 # last is sink
    
    # cut the graph
    flows,cuts = maximum_flow(gr,source,sink)
    
    # convert graph to image with labels
    res = zeros(m*n)
    for pos,label in cuts.items()[:-2]: #don't add source/sink
        res[pos] = label

    return res.reshape((m,n))
    
    
def save_as_pdf(gr,filename,show_weights=False):

    from pygraph.readwrite.dot import write
    import gv
    dot = write(gr, weighted=show_weights)
    gvv = gv.readstring(dot)
    gv.layout(gvv,'fdp')
    gv.render(gvv,'pdf',filename)


def show_labeling(im,labels):
    """    Show image with foreground and background areas.
        labels = 1 for foreground, -1 for background, 0 otherwise."""
    
    imshow(im)
    contour(labels,[-0.5,0.5])
    contourf(labels,[-1,-0.5],colors='b',alpha=0.25)
    contourf(labels,[0.5,1],colors='r',alpha=0.25)
    #axis('off')
    xticks([])
    yticks([])

