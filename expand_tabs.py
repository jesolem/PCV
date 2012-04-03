import os
import re

reg_pyfile=re.compile(r'.*\.py$',re.I)

def expand_tabs(file0):
    '''
    This function takes the name of a python source file, expands all tabs to 4 spaces (for
PEP 8 compliance), and rewrites the file in place.
    '''
    str_file_contents=open(file0,'rb').read()
    str_pep_contents=str_file_contents.replace('\x09',4*'\x20')
    open(file0,'wb').write(str_pep_contents)
    return None

def pepify_directory(path_root):
    for (path,subdir,lst_file) in os.walk(path_root):
        for file0 in (file1 for file1 in lst_file if reg_pyfile.match(file1)):
            expand_tabs(os.path.join(path,file0))
            print(os.path.join(path,file0))
            pass
        pass
    return None

if __name__=='__main__':
    pepify_directory('.')
    pass

