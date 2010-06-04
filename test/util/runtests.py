import os, sys, traceback
import unittest
import optparse

try:
    from pygame2.test.util import support, testrunner
except:
    import support, testrunner

LINEDELIM = "-" * 70
HEAVYDELIM = "=" * 70

# Excludes
EXCLUDETAGS = [ "interactive", ]

def printerror ():
    print (traceback.format_exc ())


def create_options ():
    """Create the accepatble options for the test runner."""
    optparser = optparse.OptionParser ()
    optparser.add_option ("-n", "--nosubprocess", action="store_true",
                          help="run everything in a single process "
                          "(default: use seperate subprocesses)")
    optparser.add_option ("-r", "--random", action="store_true",
                          help="randomize the order of tests")
    optparser.add_option ("-S", "--seed", type="int",
                          help="seed the randomizer (useful to "
                          "recreate earlier test cases)")
    optparser.add_option ("-i", "--interactive", action="callback",
                          callback=EXCLUDETAGS.remove,
                          callback_args=("interactive",),
                          help="also execute interactive tests ")
    return optparser

def gettestfiles (testdir=None):
    """
    Get all test files from the passed test directory. If none is
    passed, use the default pygame2 test directory.
    """
    if not testdir:
        testdir = os.path.join (os.path.dirname (__file__), "..")
        sys.path.append (testdir)

    names = os.listdir (testdir)
    testfiles = []
    for name in names:
        if name.endswith ("_test" + os.extsep + "py"):
            testfiles.append (name)
    testfiles.sort ()
    return testdir, testfiles

def loadtests (test, testdir, options):
    """Loads a test."""
    suites = []
    testloader = testrunner.TagTestLoader (EXCLUDETAGS)

    try:
        testmod = os.path.splitext (test)[0]
        glob, loc = {}, {}
        package = __import__ (testmod, glob, loc)
        print ("Loading tests from %s ..." % testmod)
        for x in dir (package):
            val = package.__dict__[x]
            if hasattr (val, "setUp") and hasattr (val, "tearDown"):
                # might be a test.
                try:
                    tests = testloader.loadTestsFromTestCase (val)
                    suites.append (tests)
                    # TODO: provide a meaningful error information about
                    # the failure.
                except:
                    printerror ()
    except:
        printerror ()
    return suites

def prepare_results (results):
    testcount = 0
    errors = []
    failures = []
    ok = 0
    for res in results:
        testcount += res.testsRun
        ok += res.testsRun - len (res.errors) - len (res.failures)
        errors.extend (res.errors)
        failures.extend (res.failures)
    return testcount, errors, failures, ok

def run ():
    optparser = create_options ()
    options, args = optparser.parse_args ()

    print (HEAVYDELIM)
    print ("-- Starting tests --")
    print (HEAVYDELIM)
    #err, out = support.redirect_output ()
    testdir, testfiles = gettestfiles ()
    testsuites = []
    for test in testfiles:
        testsuites.extend (loadtests (test, testdir, options))
    runner = testrunner.SimpleTestRunner (sys.stderr)
    
    results = []
    timetaken = 0
    print (HEAVYDELIM)
    print ("-- Executing tests --")
    print (HEAVYDELIM)
    for suite in testsuites:
        result = runner.run (suite)
        timetaken += result.duration
        results.append (result)
    print (" Finished")
    testcount, errors, failures, ok = prepare_results (results)
    print (HEAVYDELIM)
    print ("-- Statistics --")
    print (HEAVYDELIM)
    print ("Time taken:     %.3f seconds" % timetaken)
    print ("Tests executed: %d " % testcount)
    print ("Tests OK:       %d " % ok)
    print ("Tests ERROR:    %d " % len (errors))
    print ("Tests FAILURE:  %d " % len (failures))
    if len (errors) > 0:
        print ("Errors:" + os.linesep)
        for err in errors:
            print (LINEDELIM)
            print ("ERROR: " + err[0])
            print (HEAVYDELIM)
            print (err[1])
    if len (failures) > 0:
        print ("Failures:" + os.linesep)
        for fail in failures:
            print (LINEDELIM)
            print ("FAILURE: " + fail[0])
            print (HEAVYDELIM)
            print (fail[1])
        
    #support.restore_output (err, out)

if __name__ == "__main__":
    run ()
