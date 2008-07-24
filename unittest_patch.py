################################################################################

import test.unittest as unittest
import re, time, sys, StringIO
from inspect import getdoc

# This is needed for correct tracebacks
__unittest = 1

################################################################################
# Redirect stdout / stderr for the tests

def redirect_output():
    yield sys.stderr, sys.stdout
    sys.stderr, sys.stdout = StringIO.StringIO(), StringIO.StringIO()
    yield sys.stderr, sys.stdout

def restore_output(err, out):
    sys.stderr, sys.stdout = err, out

def StringIOContents(io):
    io.seek(0)
    return io.read()

################################################################################
# TestCase patching
#

def TestCase_run(self, result=None):
    if result is None: result = self.defaultTestResult()
    result.startTest(self)
    testMethod = getattr(self, self._testMethodName)
    try:
    
    ########################################################################
    # Pre run:

        #TODO: only redirect output if not tagged interactive

        result.tests[self.dot_syntax_name()] = {}
        tests = result.tests[self.dot_syntax_name()]
        (realerr, realout), (stderr, stdout) =  redirect_output()

        if 0 or 'interactive' in get_tags(testMethod):       # DEBUG
            restore_output(realerr, realout)

        t = time.time()

    ########################################################################

        for i in range(self.times_run):
            try:
                self.setUp()
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())
                return

            ok = False
            try:
                testMethod()
                ok = True
            except self.failureException:
                result.addFailure(self, self._exc_info())
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())
            
            try:
                self.tearDown()
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())
                ok = False
    
            if ok:
                if i == 0:
                    result.addSuccess(self)
            else: break

    ########################################################################
    # Post run

        t = (time.time() -t) / self.times_run
        
        restore_output(realerr, realout)

        tests["time"]   = t
        tests["stdout"] = StringIOContents(stdout)
        tests["stderr"] = StringIOContents(stderr)

    ########################################################################

    finally:
        result.stopTest(self)

################################################################################
# TestResult 
#

def TestResult___init__(self):
    self.failures   = []
    self.errors     = []
    self.tests      = {}
    self.testsRun   = 0
    self.shouldStop = 0

# TODO: all this is available in the traceback object err
FILE_LINENUMBER_RE = re.compile(r'File "([^"]+)", line ([0-9]+)')

def errorHandling(key):
    def handling(self, test, err):        
        traceback = self._exc_info_to_string(err, test)
        error_file, line_number = FILE_LINENUMBER_RE.search(traceback).groups()
        error =  (
            test.dot_syntax_name(),
            traceback,
            error_file,
            line_number,         #TODO: add locals etc
        )
        getattr(self, key).append(error)

        # Append it to individual test dict for easy access
        self.tests[test.dot_syntax_name()][key[:-1]] = error

    return handling

################################################################################

def printErrorList(self, flavour, errors):
    for test, err in ((e[0], e[1]) for e in errors):
        self.stream.writeln(self.separator1)
        self.stream.writeln("%s: %s" % (flavour, test))
        self.stream.writeln(self.separator2)
        self.stream.writeln("%s" % err)

        # DUMP REDIRECTED STDERR / STDOUT ON ERROR / FAILURE
        if self.show_redirected_on_errors:
            stderr, stdout = map(self.tests[test].get, ('stderr','stdout'))
            if stderr: self.stream.writeln("STDERR:\n%s" % stderr)
            if stdout: self.stream.writeln("STDOUT:\n%s" % stdout)

################################################################################
# Exclude by tags
#

TAGS_RE = re.compile(r"\|[tT]ags:(-?[ a-zA-Z,0-9_\n]+)\|", re.M)

class TestTags:
    def __init__(self):
        self.memoized = {}
        self.parent_modules = {}

    def get_parent_module(self, class_):
        while class_ not in self.parent_modules:
            self.parent_modules[class_] = __import__(class_.__module__)
        return self.parent_modules[class_]

    def __call__(self, obj):
        while obj not in self.memoized:
            parent_class  = obj.im_class
            parent_module = self.get_parent_module(parent_class)

            module_tags = getattr(parent_module, '__tags__', [])
            class_tags  = getattr(parent_class,  '__tags__', [])

            tags = TAGS_RE.search(getdoc(obj) or '')
            if tags: test_tags = [t.strip() for t in tags.group(1).split(',')]
            else:    test_tags = []
        
            combined = set()
            for tags in (module_tags, class_tags, test_tags):
                if not tags: continue
        
                add    = set(t for t in tags if not t.startswith('-'))
                remove = set(t[1:] for t in tags if t not in add)
        
                if add:     combined.update(add)
                if remove:  combined.difference_update(remove)
    
            self.memoized[obj] = combined

        return self.memoized[obj]

get_tags = TestTags()

################################################################################

def getTestCaseNames(self, testCaseClass):
    def test_wanted(attrname, testCaseClass=testCaseClass,
                              prefix=self.testMethodPrefix):
        if not attrname.startswith(prefix): return False
        else:
            actual_attr = getattr(testCaseClass, attrname)
            return (
                 callable(actual_attr) and
                 not [t for t in  get_tags(actual_attr) if t in self.exclude]
            )

    # TODO:

    # Replace test_not_implemented mechanism with technique that names the tests
    # todo_test_xxxxxx, then when wanting to fail them, loads any members that
    # startswith(test_prefix)
    
    # REGEX FOR TEST_NOT_IMPLEMENTED
    # SEARCH:
    #    def (test_[^ ]+)((?:\s+#.*\n?)+\s+)self\.assert_\(test_not_implemented\(\)\)
    # REPLACE:
    #    def todo_\1\2self.fail()
    
    testFnNames = filter(test_wanted, dir(testCaseClass))

    for baseclass in testCaseClass.__bases__:
        for testFnName in self.getTestCaseNames(baseclass):
            if testFnName not in testFnNames:  # handle overridden methods
                testFnNames.append(testFnName)

    if self.sortTestMethodsUsing:
        testFnNames.sort(self.sortTestMethodsUsing)

    return testFnNames

################################################################################

def patch(options):
    if options.incomplete:
        unittest.TestLoader.testMethodPrefix = (
            unittest.TestLoader.testMethodPrefix, 'todo_'
        )

    unittest.TestLoader.getTestCaseNames = getTestCaseNames
    unittest.TestLoader.exclude = (
        [e.strip() for e in options.exclude.split(',')] )

    # Timing
    unittest.TestCase.times_run = options.timings
    unittest.TestCase.run = TestCase_run
    unittest.TestCase.dot_syntax_name = lambda self: (
        "%s.%s"% (self.__class__.__name__, self._testMethodName) )

    # Error logging
    unittest.TestResult.show_redirected_on_errors = options.show_output
    unittest.TestResult.__init__   = TestResult___init__
    unittest.TestResult.addError   = errorHandling('errors')
    unittest.TestResult.addFailure = errorHandling('failures')

    unittest._TextTestResult.printErrorList = printErrorList
    
################################################################################