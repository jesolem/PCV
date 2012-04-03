from numpy import *
import pickle
from pysqlite2 import dbapi2 as sqlite


class Indexer(object):
    
    def __init__(self,db,voc):
        """ Initialize with the name of the database 
            and a vocabulary object. """
            
        self.con = sqlite.connect(db)
        self.voc = voc
    
    def __del__(self):
        self.con.close()
    
    def db_commit(self):
        self.con.commit()
    
    def get_id(self,imname):
        """ Get an entry id and add if not present. """
        
        cur = self.con.execute(
        "select rowid from imlist where filename='%s'" % imname)
        res=cur.fetchone()
        if res==None:
            cur = self.con.execute(
            "insert into imlist(filename) values ('%s')" % imname)
            return cur.lastrowid
        else:
            return res[0] 
    
    def is_indexed(self,imname):
        """ Returns True if imname has been indexed. """
        
        im = self.con.execute("select rowid from imlist where filename='%s'" % imname).fetchone()
        return im != None
    
    def add_to_index(self,imname,descr):
        """ Take an image with feature descriptors, 
            project on vocabulary and add to database. """
            
        if self.is_indexed(imname): return
        print 'indexing', imname
        
        # get the imid
        imid = self.get_id(imname)
        
        # get the words
        imwords = self.voc.project(descr)
        nbr_words = imwords.shape[0]
        
        # link each word to image
        for i in range(nbr_words):
            word = imwords[i]
            # wordid is the word number itself
            self.con.execute("insert into imwords(imid,wordid,vocname) values (?,?,?)", (imid,word,self.voc.name))
            
        # store word histogram for image
        # use pickle to encode NumPy arrays as strings
        self.con.execute("insert into imhistograms(imid,histogram,vocname) values (?,?,?)", (imid,pickle.dumps(imwords),self.voc.name))
    
    def create_tables(self): 
        """ Create the database tables. """
        
        self.con.execute('create table imlist(filename)')
        self.con.execute('create table imwords(imid,wordid,vocname)')
        self.con.execute('create table imhistograms(imid,histogram,vocname)')        
        self.con.execute('create index im_idx on imlist(filename)')
        self.con.execute('create index wordid_idx on imwords(wordid)')
        self.con.execute('create index imid_idx on imwords(imid)')
        self.con.execute('create index imidhist_idx on imhistograms(imid)')
        self.db_commit()


class Searcher(object):
    
    def __init__(self,db,voc):
        """ Initialize with the name of the database. """
        self.con = sqlite.connect(db)
        self.voc = voc
    
    def __del__(self):
        self.con.close()
    
    def get_imhistogram(self,imname):
        """ Return the word histogram for an image. """
        
        im_id = self.con.execute(
            "select rowid from imlist where filename='%s'" % imname).fetchone()
        s = self.con.execute(
            "select histogram from imhistograms where rowid='%d'" % im_id).fetchone()
        
        # use pickle to decode NumPy arrays from string
        return pickle.loads(str(s[0]))
    
    def candidates_from_word(self,imword):
        """ Get list of images containing imword. """
        
        im_ids = self.con.execute(
            "select distinct imid from imwords where wordid=%d" % imword).fetchall()
        return [i[0] for i in im_ids]
    
    def candidates_from_histogram(self,imwords):
        """ Get list of images with similar words. """
        
        # get the word ids
        words = imwords.nonzero()[0]
        
        # find candidates
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates+=c
        
        # take all unique words and reverse sort on occurrence 
        tmp = [(w,candidates.count(w)) for w in set(candidates)]
        tmp.sort(cmp=lambda x,y:cmp(x[1],y[1]))
        tmp.reverse()
        
        # return sorted list, best matches first    
        return [w[0] for w in tmp] 
    
    def query(self,imname):
        """ Find a list of matching images for imname. """
        
        h = self.get_imhistogram(imname)
        candidates = self.candidates_from_histogram(h)
        
        matchscores = []
        for imid in candidates:
            # get the name
            cand_name = self.con.execute(
                "select filename from imlist where rowid=%d" % imid).fetchone()
            cand_h = self.get_imhistogram(cand_name)
            cand_dist = sqrt( sum( self.voc.idf*(h-cand_h)**2 ) )
            matchscores.append( (cand_dist,imid) )
        
        # return a sorted list of distances and database ids
        matchscores.sort()
        return matchscores
    
    def get_filename(self,imid):
        """ Return the filename for an image id. """
        
        s = self.con.execute(
            "select filename from imlist where rowid='%d'" % imid).fetchone()
        return s[0]


def tf_idf_dist(voc,v1,v2):
    
    v1 /= sum(v1)
    v2 /= sum(v2)
    
    return sqrt( sum( voc.idf*(v1-v2)**2 ) )


def compute_ukbench_score(src,imlist):
    """ Returns the average number of correct
        images on the top four results of queries. """
        
    nbr_images = len(imlist)
    pos = zeros((nbr_images,4))
    # get first four results for each image
    for i in range(nbr_images):
        pos[i] = [w[1]-1 for w in src.query(imlist[i])[:4]]
    
    # compute score and return average
    score = array([ (pos[i]//4)==(i//4) for i in range(nbr_images)])*1.0
    return sum(score) / (nbr_images)


# import PIL and pylab for plotting        
from PIL import Image
from pylab import *

def plot_results(src,res):
    """ Show images in result list 'res'. """
    
    figure()
    nbr_results = len(res)
    for i in range(nbr_results):
        imname = src.get_filename(res[i])
        subplot(1,nbr_results,i+1)
        imshow(array(Image.open(imname)))
        axis('off')
    show()