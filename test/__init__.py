import sys

#
# Python 3.x gets weird hickups with the module names, if we try to import
# run_tests.run() directly.
#
if sys.version_info[0] < 3:
    from pygame2.test.run_tests import run
else:
    def run ():
        import pygame2.test.run_tests
        pygame2.test.run_tests.run ()
