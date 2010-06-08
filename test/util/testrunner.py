##
## This file is placed under the public domain.
##

import sys, os
import unittest
from unittest import TestResult, TestLoader
import time

class TagTestLoader (TestLoader):
    """A TestLoader which handles additional __tags__ attributes for
    test functions.
    """
    def __init__ (self, excludetags, randomizer=None):
        TestLoader.__init__ (self)
        self.excludetags = excludetags
        self.randomizer = randomizer

    def getTestCaseNames(self, testCaseClass):
        """
        Gets only the tests, which are not within the tag exclusion.
        The method overrides the original TestLoader.getTestCaseNames()
        method, so we need to keep them in sync on updates.
        """
        def isTestMethod(attrname, testCaseClass=testCaseClass,
                         prefix=self.testMethodPrefix):
            if not attrname.startswith (prefix):
                return False
            if not hasattr (getattr (testCaseClass, attrname), '__call__'):
                return False
            if hasattr (getattr (testCaseClass, attrname), "__tags__"):
                # Tagged test method
                tags = getattr (getattr (testCaseClass, attrname), "__tags__")
                for t in tags:
                    if t in self.excludetags:
                        return False
            return True

        if hasattr (testCaseClass, "__tags__"):
            tags = getattr (testCaseClass, "__tags__")
            for t in tags:
                if t in self.excludetags:
                    return []

        testFnNames = list (filter(isTestMethod, dir(testCaseClass)))
        cmpkey = None
        if hasattr (unittest, "_CmpToKey"):
            cmpkey = unittest._CmpToKey
        elif hasattr (unittest, "CmpToKey"):
            cmpkey = unittest.CmpToKey

        if self.randomizer:
            self.randomizer.shuffle (testFnNames)
        elif self.sortTestMethodsUsing:
            if cmpkey:
                testFnNames.sort (key=cmpkey(self.sortTestMethodsUsing))
            else:
                testFnNames.sort ()
        return testFnNames

class SimpleTestResult (TestResult):
    """A simple TestResult class with output capabilities.
    """
    def __init__ (self, stream=sys.stderr, verbose=False, countcall=None):
        TestResult.__init__ (self)
        self.stream = stream
        self.duration = 0
        self.verbose = verbose
        self.countcall = countcall
    
    def addSuccess (self, test):
        TestResult.addSuccess (self, test)
        if self.verbose:
            self.stream.write ("OK:     %s%s" % (test, os.linesep))
            self.stream.flush ()
        self.countcall ()

    def addError (self, test, err):
        TestResult.addError (self, test, err)
        if self.verbose:
            self.stream.write ("ERROR:  %s%s" % (test, os.linesep))
            self.stream.flush ()
        self.countcall ()

    def addFailure (self, test, err):
        TestResult.addFailure (self, test, err)
        if self.verbose:
            self.stream.write ("FAILED: %s%s" % (test, os.linesep))
            self.stream.flush ()
        self.countcall ()

class SimpleTestRunner (object):
    def __init__ (self, stream=sys.stderr, verbose=False):
        self.stream = stream
        self.verbose = verbose

    def run (self, test, countcall):
        result = SimpleTestResult (self.stream, self.verbose, countcall)
        starttime = time.time ()
        test (result)
        endtime = time.time ()
        result.duration = endtime - starttime
        return result
