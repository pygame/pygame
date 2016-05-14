################################################################################

if __name__.startswith('pygame.tests.'):
    from pygame.tests.test_utils import unittest, import_submodule
else:
    from test.test_utils import unittest, import_submodule
import re
import time
import sys
try: 
    import StringIO
except ImportError:
    import io as StringIO
import random

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

        result.tests[self.dot_syntax_name()] = {
            'times' : [],
        }
        
        tests = result.tests[self.dot_syntax_name()]
        (realerr, realout), (stderr, stdout) =  redirect_output()
        test_tags = list(get_tags(self.__class__, testMethod))

        if 0 or 'interactive' in test_tags:       # DEBUG
            restore_output(realerr, realout)

    ########################################################################

        for i in range(self.times_run):
            t = time.time()
            
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
            
            
            tests["times"] += [time.time() -t]
            
            if not ok: break            
        
            # if ok:
            #     if i == 0:
            #         result.addSuccess(self)
            # else: break
    
        if ok:
            result.addSuccess(self)
    ########################################################################
    # Post run

        restore_output(realerr, realout)

        tests["stdout"] = stdout.getvalue()
        tests["stderr"] = stderr.getvalue()
        tests["tags"]   = test_tags

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
    for test, err in [(e[0], e[1]) for e in errors]:
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
        if class_ not in self.parent_modules:
            self.parent_modules[class_] = import_submodule(class_.__module__)
        return self.parent_modules[class_]

    def __call__(self, parent_class, meth):
        key = (parent_class, meth.__name__)
        if key not in self.memoized:
            parent_module = self.get_parent_module(parent_class)

            module_tags = getattr(parent_module, '__tags__', [])
            class_tags  = getattr(parent_class,  '__tags__', [])

            tags = TAGS_RE.search(getdoc(meth) or '')
            if tags: test_tags = [t.strip() for t in tags.group(1).split(',')]
            else:    test_tags = []
        
            combined = set()
            for tags in (module_tags, class_tags, test_tags):
                if not tags: continue
        
                add    = set([t for t in tags if not t.startswith('-')])
                remove = set([t[1:] for t in tags if t not in add])
        
                if add:     combined.update(add)
                if remove:  combined.difference_update(remove)
    
            self.memoized[key] = combined

        return self.memoized[key]

get_tags = TestTags()

################################################################################
# unittest.TestLoader
#
def CmpToKey(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) == -1
    return K

def getTestCaseNames(self, testCaseClass):
    def test_wanted(attrname, testCaseClass=testCaseClass,
                              prefix=self.testMethodPrefix):
        if not attrname.startswith(prefix): return False
        else:
            actual_attr = getattr(testCaseClass, attrname)
            return (
                 hasattr(actual_attr, '__call__') and
                 not [t for t in  get_tags(testCaseClass, actual_attr)
                      if t in self.exclude]
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

    testFnNames = [c for c in dir(testCaseClass) if test_wanted(c)]

    for baseclass in testCaseClass.__bases__:
        for testFnName in self.getTestCaseNames(baseclass):
            if testFnName not in testFnNames:  # handle overridden methods
                testFnNames.append(testFnName)

    if self.randomize_tests: 
        random.shuffle(testFnNames)
    elif self.sortTestMethodsUsing:
        testFnNames.sort(key=CmpToKey(self.sortTestMethodsUsing))

    return testFnNames

################################################################################

def patch(**kwds):
    """Customize the unittest module according to the run-time options

    Recognized keyword arguments:
    incomplete, randomize, seed, exclude, timings and show_output
    
    """

    option_incomplete = kwds.get('incomplete', False)
    option_randomize = kwds.get('randomize', False)
    try:
        option_seed = kwds['seed'] is not None
    except KeyError:
        option_seed = False
    option_exclude = kwds.get('exclude', ('interactive',))
    option_timings = kwds.get('timings', 1)
    option_show_output = kwds.get('show_output', False)

    # Incomplete todo_xxx tests
    if option_incomplete:
        unittest.TestLoader.testMethodPrefix = (
            unittest.TestLoader.testMethodPrefix, 'todo_'
        )
    
    # Randomizing    
    unittest.TestLoader.randomize_tests = option_randomize or option_seed

    unittest.TestLoader.getTestCaseNames = getTestCaseNames
    unittest.TestLoader.exclude = option_exclude

    # Timing
    unittest.TestCase.times_run = option_timings
    unittest.TestCase.run = TestCase_run
    unittest.TestCase.dot_syntax_name = lambda self: (
        "%s.%s"% (self.__class__.__name__, self._testMethodName) )

    # Error logging
    unittest.TestResult.show_redirected_on_errors = option_show_output
    unittest.TestResult.__init__   = TestResult___init__
    unittest.TestResult.addError   = errorHandling('errors')
    unittest.TestResult.addFailure = errorHandling('failures')

    unittest._TextTestResult.printErrorList = printErrorList
    
################################################################################

