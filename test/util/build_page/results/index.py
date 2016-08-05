#!/bin/sh
""":"
PYTHONPATH=/home/wazoocom/www/pygame/libs exec /home/wazoocom/bin/python $0 ${1+"$@"}
"""

from __future__ import with_statement

import os

from pywebsite import *
from helpers import *

def main():
    q = cgi.FieldStorage()
    
    platform_id = q['platform_id'].value
    revision = q['revision'].value
    which_info = q.has_key('info') and q['info'].value.strip() or 'setup'
    
    results_zip = '%s_%s.zip' % (platform_id, revision)
    
    rz = ResultsZip('%s/archives/%s' % (platform_id, results_zip) )
    
    tabs, tab_contents = rz.html(which_info)
    
    TEMPLATE = """
    <html>
    
    <head>
        <meta http-equiv="Content-Type" content="text/html" charset="utf-8" />
    
        <title> Build Info for %(platform_id)s </title>    
    
        <link href="/results/results.css" rel="stylesheet" type="text/css"></link>

    </head>
    
    <body>
        <h1> <a href="/results/%(platform_id)s/archives/%(results_zip)s"> %(results_zip)s </a> </h1> 
    
        <p><a href="/index.php"> builds </a></p>
    
        <div class="tabbed">
            
            <ul class="tabs"> 
                
                %(tabs)s
            
            </ul>
            
                %(tab_contents)s
    
        </div>
    </body>
    </html>
    """

    print TEMPLATE % locals()

try:
    main()
except Exception, e:
    print 'woah woah!'