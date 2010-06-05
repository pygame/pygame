import os, sys, traceback
import unittest
import optparse
import random

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

def include_tag (option, opt, value, parser, *args, **kwargs):
    try:
        if args:
            EXCLUDETAGS.remove (args[0])
        else:
            EXCLUDETAGS.remove (value)
    finally:
        pass

def exclude_tag (option, opt, value, parser, *args, **kwargs):
    if value not in EXCLUDETAGS:
        EXCLUDETAGS.append (value)

def create_options ():
    """Create the accepatble options for the test runner."""
    optparser = optparse.OptionParser ()
    optparser.add_option ("-s", "--subprocess", action="store_true",
                          default=False,
                          help="run everything in an own subprocess "
                          "(default: use a single process)")
    optparser.add_option ("-v", "--verbose", action="store_true", default=False,
                          help="be verbose adnd print anything instantly")
    optparser.add_option ("-r", "--random", action="store_true", default=False,
                          help="randomize the order of tests")
    optparser.add_option ("-S", "--seed", type="int",
                          help="seed the randomizer (useful to "
                          "recreate earlier randomized test cases)")
    optparser.add_option ("-i", "--interactive", action="callback",
                          callback=include_tag,
                          callback_args=("interactive",),
                          help="also execute interactive tests")
    optparser.add_option ("-e", "--exclude", action="callback",
                          callback=exclude_tag, type="string",
                          help="exclude test containing the tag")
    optkeys = [
        "subprocess",
        "random",
        "seed",
        "verbose"
        ]

    return optparser, optkeys

def gettestfiles (testdir=None, randomizer=None):
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
    if randomizer:
        randomizer.shuffle (testfiles)
    else:
        testfiles.sort ()
    return testdir, testfiles

def loadtests (test, testdir, writer, loader, options):
    """Loads a test."""
    suites = []

    try:
        testmod = os.path.splitext (test)[0]
        glob, loc = {}, {}
        package = __import__ (testmod, glob, loc)
        if options.verbose:
            writer.writeline ("Loading tests from [%s] ..." % testmod)
        else:
            writer.writesame ("Loading tests from [%s] ..." % testmod)
        for x in dir (package):
            val = package.__dict__[x]
            if hasattr (val, "setUp") and hasattr (val, "tearDown"):
                # might be a test.
                try:
                    tests = loader.loadTestsFromTestCase (val)
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
    optparser, optkeys = create_options ()
    options, args = optparser.parse_args ()
    #err, out = support.redirect_output ()
    writer = support.StreamOutput (sys.stdout)

    if options.verbose:
        writer.writeline (HEAVYDELIM)
        writer.writeline ("-- Starting tests --")
        writer.writeline (HEAVYDELIM)

    loader = None
    randomizer = None
    if options.random:
        if options.seed is None:
            options.seed = random.randint (0, sys.maxint)
        randomizer = random.Random (options.seed)
    loader = testrunner.TagTestLoader (EXCLUDETAGS, randomizer)

    testdir, testfiles = gettestfiles (randomizer=randomizer)
    testsuites = []
    for test in testfiles:
        testsuites.extend (loadtests (test, testdir, writer, loader, options))
    if not options.verbose:
        writer.writesame ("Tests loaded")
    runner = testrunner.SimpleTestRunner (sys.stderr, options.verbose)
    
    results = []
    timetaken = 0

    if options.verbose:
        writer.writeline (HEAVYDELIM)
        writer.writeline ("-- Executing tests --")
        writer.writeline (HEAVYDELIM)

    maxcount = 0
    curcount = 0
    for suite in testsuites:
        maxcount += suite.countTestCases ()

    class writerunning:
        def __init__ (self, maxcount, verbose):
            self.curcount = 0
            self.maxcount = maxcount
            self.verbose = verbose

        def __call__ (self):
            self.curcount += 1
            if not self.verbose:
                writer.writesame ("Running tests [ %d / %d ] ..." %
                                  (self.curcount, self.maxcount))

    runwrite = writerunning (maxcount, options.verbose)

    for suite in testsuites:
        result = runner.run (suite, runwrite)
        timetaken += result.duration
        curcount += result.testsRun
        results.append (result)
    writer.writeline ()
    testcount, errors, failures, ok = prepare_results (results)

    writer.writeline (HEAVYDELIM)
    writer.writeline ("-- Statistics --")
    writer.writeline (HEAVYDELIM)
    writer.writeline ("Options:")
    for key in optkeys:
        writer.writeline ("                '%s' = '%s'" %
                          (key, getattr (options, key)))
    writer.writeline ("Time taken:     %.3f seconds" % timetaken)
    writer.writeline ("Tests executed: %d " % testcount)
    writer.writeline ("Tests OK:       %d " % ok)
    writer.writeline ("Tests ERROR:    %d " % len (errors))
    writer.writeline ("Tests FAILURE:  %d " % len (failures))
    
    if len (errors) > 0:
        writer.writeline ("Errors:" + os.linesep)
        for err in errors:
            writer.writeline (LINEDELIM)
            writer.writeline ("ERROR: %s" % err[0])
            writer.writeline (HEAVYDELIM)
            writer.writeline (err[1])
    if len (failures) > 0:
        writer.writeline ("Failures:" + os.linesep)
        for fail in failures:
            writer.writeline (LINEDELIM)
            writer.writeline ("FAILURE: %s" % fail[0])
            writer.writeline (HEAVYDELIM)
            writer.writeline (fail[1])
        
    #support.restore_output (err, out)

if __name__ == "__main__":
    run ()
