import sys, os
import unittest
from unittest import TestResult, TestLoader
import time

class TagTestLoader (TestLoader):

    def __init__ (self, excludetags):
        TestLoader.__init__ (self)
        self.excludetags = excludetags

    def getTestCaseNames(self, testCaseClass):
        """Return a sorted sequence of method names found within testCaseClass
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
        
        testFnNames = list (filter(isTestMethod, dir(testCaseClass)))
        cmpkey = None
        if hasattr (unittest, "_CmpToKey"):
            cmpkey = unittest._CmpToKey
        elif hasattr (unittest, "CmpToKey"):
            cmpkey = unittest.CmpToKey

        if self.sortTestMethodsUsing:
            if cmpkey:
                testFnNames.sort (key=cmpkey(self.sortTestMethodsUsing))
            else:
                testFnNames.sort ()
        return testFnNames

class SimpleTestResult (TestResult):
    def __init__ (self, stream=sys.stderr):
        TestResult.__init__ (self)
        self.stream = stream
        self.duration = 0
    
    def addSuccess (self, test):
        TestResult.addSuccess (self, test)
        self.stream.write (".")
        self.stream.flush ()

    def addError (self, test, err):
        TestResult.addError (self, test, err)
        self.stream.write ("E")
        self.stream.flush ()

    def addFailure (self, test, err):
        TestResult.addFailure (self, test, err)
        self.stream.write ("F")
        self.stream.flush ()

class SimpleTestRunner (object):
    def __init__ (self, stream=sys.stderr):
        self.stream = stream

    def run (self, test):
        result = SimpleTestResult (self.stream)
        starttime = time.time ()
        test (result)
        endtime = time.time ()
        result.duration = endtime - starttime
        return result
