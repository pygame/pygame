# package trackmod

"""A package for tracking module use

Exports begin(repfilepth=None).

"""

from trackmod import reporter  # Want this first
import sys
import atexit

from trackmod import importer

try:
    installed
except NameError:
    installed = False

def generate_report(repfilepth):
    try:
        repfile = open(repfilepth, 'w')
    except:
        return
    try:
        reporter.write_report(repfile)
    finally:
        repfile.close()
    
def begin(repfilepth=None):
    global installed

    if not installed:
        sys.meta_path.insert(0, importer)
        installed = True
        if repfilepth is not None:
            atexit.register(generate_report, repfilepth)

def end():
    reporter.end()
    importer.end()

reporter.begin()  # Keep this last.



