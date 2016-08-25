# -*- coding:utf-8 -*-

import os
import shutil
import tempfile
from six.moves import urllib


def maybe_create_dir(dirname):
    assert( os.path.exists(dirname) == False or os.path.isdir(dirname) )
    curdir = ''
    for d in os.path.split(dirname):
        curdir = os.path.join(curdir, d)
        if os.path.exists(curdir) == False:
            os.mkdir(curdir)
    
def maybe_download(filename, url):
    if os.path.exists(filename):
        return filename
    maybe_create_dir( os.path.dirname(filename) )
    with tempfile.NamedTemporaryFile() as tmpfile:
        print( "downloading from {} ...".format(url) )
        urllib.request.urlretrieve(url, tmpfile.name)
        shutil.copyfile(tmpfile.name, filename)
        print( "saved as {}.".format(filename) )        
    return filename
