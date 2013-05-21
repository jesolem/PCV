import cherrypy, os, urllib, pickle
from numpy import *

from PCV.imagesearch import imagesearch

"""
This is the image search demo in Section 7.6.
"""

class SearchDemo:
    
    def __init__(self):
        # load list of images
        f = open('webimlist.txt')
        self.imlist = f.readlines()
        f.close()
        self.nbr_images = len(self.imlist)
        self.ndx = range(self.nbr_images)
        
        # load vocabulary
        f = open('vocabulary.pkl', 'rb')
        self.voc = pickle.load(f)
        f.close()
        
        # set max number of results to show
        self.maxres = 15
        
        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>Image search example</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """
        
    def index(self,query=None):
        self.src = imagesearch.Searcher('web.db',self.voc)
        
        html = self.header
        html += """
            <br />
            Click an image to search. <a href='?query='> Random selection </a> of images.
            <br /><br />
            """
        if query:
            # query the database and get top images
            res = self.src.query(query)[:self.maxres]
            for dist,ndx in res:
                imname = self.src.get_filename(ndx)
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='100' />"
                html += "</a>"
        else:
            # show random selection if no query
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='100' />"
                html += "</a>"
                
        html += self.footer
        return html
    
    index.exposed = True

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))