"""Unit tests for Pygame2.

This package contains the unit tests for Pygame2. You can execute the
unit tests using

    python -m pygame2.test

"""
import sys

#
# Python 3.x gets weird hickups with the module names, if we try to import
# run_tests.run() directly.
#
#if sys.version_info[0] < 3:
from pygame2.test.util.runtests import run
#else:
#    def run ():
#        import pygame2.test.util.run_tests
#       pygame2.test.run_tests.run ()

if __name__ == "__main__":
    run ()
