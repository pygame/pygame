#!/usr/bin/env python
import sys, os, re, unittest

main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
test_subdir = 'test'

# Make sure we're in the correct directory
os.chdir( main_dir )

# Add the modules directory to the python path    
sys.path.insert( 0, test_subdir )

# Load test util functions
import test_utils

# Load all the tests
suite = unittest.TestSuite()
test_module_re = re.compile('^(.+_test)\.py$')
for file in os.listdir(test_subdir):
    for module in test_module_re.findall(file):
        if module == "scrap_test":
            continue
        print 'loading ' + module
        __import__( module )
        test = unittest.defaultTestLoader.loadTestsFromName( module )
        suite.addTest( test )

# Parse command line options
if "--incomplete" in sys.argv or "-i" in sys.argv:
    test_utils.fail_incomplete_tests = 1

verbose = "--verbose" in sys.argv or "-v" in sys.argv

# Run the tests
runner = unittest.TextTestRunner()

if verbose: runner.verbosity = 2
runner.run( suite )