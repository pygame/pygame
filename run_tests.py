#!/usr/bin/env python
import sys, os, re, unittest

main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
test_subdir = 'test'

# Make sure we're in the correct directory
os.chdir( main_dir )

# Add the modules directory to the python path    
sys.path.insert( 0, test_subdir )

# Load all the tests
suite = unittest.TestSuite()
test_module_re = re.compile('^(.+_test)\.py$')
for file in os.listdir(test_subdir):
    for module in test_module_re.findall(file):
        print 'loading ' + module
        __import__( module )
        test = unittest.defaultTestLoader.loadTestsFromName( module )
        suite.addTest( test )

# Run the tests
runner = unittest.TextTestRunner()
runner.run( suite )
