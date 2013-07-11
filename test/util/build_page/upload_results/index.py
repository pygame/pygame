#!/bin/sh
""":"
PYTHONPATH=/home/wazoocom/www/pygame/libs exec /home/wazoocom/bin/python $0 ${1+"$@"}
"""

################################################################################

# Std Libs
import os

# User Libs
from pywebsite import *
from helpers import relative_to, ResultsZip

import process_results

################################################################################

def prepare_results_dirs(d):
    archives_dir =  os.path.join(d, 'archives')
    db_dir = os.path.join(d, 'db')
    
    for dir in (archives_dir, db_dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

################################################################################

q = cgi.FieldStorage()

if q.has_key('results_file'):
    rz = ResultsZip(q['results_file'].file)

    results_dir = relative_to(__file__, '../results/%s' % rz.platform_id)
    prepare_results_dirs(results_dir)

    if 'test_results_dict.txt' in rz.namelist():
        db = ZDB(os.path.join(results_dir, 'db', '%s.fs' % rz.platform_id))
        db.root[rz.latest_rev] = rz.eval('test_results_dict.txt')
        db.close()

    process_results.process_zip(rz, results_dir)
    
    rz.close()

    print '%s RESULTS UPLOAD SUCCESSFUL' % rz.platform_id.upper()
else:
    print """
    <html>

    <head><title>Testing</title></head>
    
    <body>
    
    <form enctype="multipart/form-data" action="index.py" method="POST">
        <input type="hidden" name="MAX_FILE_SIZE" value="30000">
    
        Send this file: <input name="results_file" type="file">
    
        <input type="submit" value="Send File">
    </form>
    
    </body>
    
    </html>"""

################################################################################