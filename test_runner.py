################################################################################

import sys, os, re, unittest, StringIO, time, optparse
from inspect import getdoc, getmembers, isclass
from pprint import pformat

################################################################################

def prepare_test_env():
    main_dir = os.path.split(os.path.abspath(__file__))[0]
    test_subdir = os.path.join(main_dir, 'test')
    sys.path.insert(0, test_subdir)
    fake_test_subdir = os.path.join(test_subdir, 'run_tests__tests')
    return main_dir, test_subdir, fake_test_subdir

main_dir, test_subdir, fake_test_subdir = prepare_test_env()
import test_utils

################################################################################
# Set the command line options
#
# options are shared with run_tests.py so make sure not to conflict
# in time more will be added here

opt_parser = optparse.OptionParser()

opt_parser.add_option (
     "-i",  "--incomplete", action = 'store_true',
     help   = "fail incomplete tests" )

opt_parser.add_option (
     "-s",  "--subprocess", action = 'store_true',
     help   = "run test suites in subprocesses (default: same process)" )

opt_parser.add_option (
     "-d",  "--dump", action = 'store_true',
     help   = "dump failures/errors as dict ready to eval" )

opt_parser.add_option (
     "-e",  "--exclude", 
     help   = "exclude tests containing any of TAGS" )

opt_parser.add_option (
     "-a",  "--all", action = 'store_true',
     help   = "dump all results not just errors eg. -da" )

opt_parser.add_option (
     "-H",  "--human", action = 'store_true',
     help   = "dump results as dict ready to eval if unsure "
              "that pieced together results are correct "
              "(subprocess mode)" ) # TODO

opt_parser.add_option (
     "-m",  "--multi_thread", metavar = 'THREADS', type = 'int',
     help   = "run subprocessed tests in x THREADS" )

opt_parser.add_option (
     "-t",  "--time_out", metavar = 'SECONDS', type = 'int',
     help   = "kill stalled subprocessed tests after SECONDS" )

opt_parser.add_option (
     "-f",  "--fake", metavar = "DIR",
     help   = "run fake tests in run_tests__tests/$DIR" )

opt_parser.add_option (
     "-p",  "--python", metavar = "PYTHON",
     help   = "path to python excutable to run subproccesed tests\n"
              "default (sys.executable): %s" % sys.executable)

################################################################################

TEST_RESULTS_START = "<--!! TEST RESULTS START HERE !!-->"
TEST_RESULTS_RE = re.compile('%s\n(.*)' % TEST_RESULTS_START, re.DOTALL | re.M)
FILE_LINENUMBER_RE = re.compile(r'File "([^"]+)", line ([0-9]+)')

def get_test_results(raw_return):
    test_results = TEST_RESULTS_RE.search(raw_return)
    if test_results:
        try:     return eval(test_results.group(1))
        except:  raise Exception (
            "BUGGY TEST RESULTS EVAL:\n %s" % test_results.group(1)
        )

def count(results, *args, **kw):
    if kw.get('single'): results = {'single' : results}
    for arg in args:
        all_of = [a for a in [v.get(arg) for v in results.values()] if a]
        if not all_of: yield 0
        else:
            if isinstance(all_of[0], int): the_sum = all_of
            else: the_sum = (len(v) for v in all_of)
            yield sum(the_sum)

################################################################################

def redirect_output():
    yield sys.stderr, sys.stdout
    sys.stderr, sys.stdout = StringIO.StringIO(), StringIO.StringIO()
    yield sys.stderr, sys.stdout

def restore_output(err, out):
    sys.stderr, sys.stdout = err, out

def StringIOContents(io):
    io.seek(0)
    return io.read()

def merged_dict(*args):
    dictionary = {}
    for arg in args: dictionary.update(arg)        
    return dictionary
    
def from_namespace(ns, listing):
    return dict((i, ns[i]) for i in listing)

def many_modules_key(modules):
    return ', '.join(modules)

################################################################################
# ERRORS

unittest._TextTestResult.monkeyRepr = lambda self, flavour, errors:  [
    (
        "%s: %s" % (flavour, self.getDescription(e[0])),     # Description
        e[1],                                                # TraceBack
        FILE_LINENUMBER_RE.search(e[1]).groups(),            # Blame Info
    )
    for e in errors
]

def make_complete_failure_error(result):
    return (
        "ERROR: all_tests_for (%s.AllTestCases)" % result['module'],
        "Complete Failure (ret code: %s)" % result['return_code'],
        (result['test_file'], '1'),
    )

def combined_errs(results):
    for result in results.itervalues():
        combined_errs = result['errors'] + result['failures']
        for err in combined_errs:
            yield err

# For combined results, plural, used in subprocess mode
def test_failures(results):
    errors = {}
    total, = count(results, 'num_tests')

    for module, result in results.items():
        num_errors = sum(count(result, 'failures', 'errors', single = 1))
        if num_errors is 0 and result['return_code']:
            result.update(RESULTS_TEMPLATE)
            result['errors'].append(make_complete_failure_error(result))
            num_errors += 1
        if num_errors: errors.update({module:result})

    return total, errors

################################################################################

TAGS_RE = re.compile(r"\|[tT]ags:([ a-zA-Z,0-9_\n]+)\|", re.DOTALL | re.MULTILINE)

def get_tags(obj):
    tags = TAGS_RE.search(getdoc(obj) or '')
    return tags and [t.strip() for t in tags.group(1).split(',')] or []

def is_test_case(obj):
    return isclass(obj) and issubclass(obj, unittest.TestCase)

def is_test(obj):
    return callable(obj) and obj.__name__.startswith('test_') 

def filter_by_tags(module, tags):
    for tcstr, test_case in (m for m in getmembers(module, is_test_case)):
        for tstr, test in (t for t in getmembers(test_case, is_test)):
            for tag in get_tags(test):
                if tag in tags:
                    exec 'del module.%s.%s' % (tcstr, tstr)
                    break

################################################################################
# For complete failures (+ namespace saving)

RESULTS_TEMPLATE = {
    'output'     :  '',
    'stderr'     :  '',
    'stdout'     :  '',
    'num_tests'  :   0,
    'failures'   :  [],
    'errors'     :  [],
}

################################################################################

def run_test(modules, options):
    if isinstance(modules, str): modules = [modules]
    suite = unittest.TestSuite()

    #TODO: ability to pass module.TestCase etc (names) from run_test.py
    for module in modules:
        m = __import__(module)

        print 'loading', module

        if options.exclude:
            filter_by_tags(m, [e.strip() for e in options.exclude.split(',')])
        
        # decorate tests with profiling wrappers etc
        
        test = unittest.defaultTestLoader.loadTestsFromName(module)
        suite.addTest(test)

    (realerr, realout), (stderr, stdout) =  redirect_output()
    # restore_output(realerr, realout)       DEBUG

    output = StringIO.StringIO()
    runner = unittest.TextTestRunner(stream = output)

    test_utils.fail_incomplete_tests = options.incomplete

    results = runner.run(suite)
    output, stderr, stdout = map(StringIOContents, (output, stderr, stdout))
    restore_output(realerr, realout)
    
    num_tests = results.testsRun
    failures  = results.monkeyRepr('FAIL', results.failures)
    errors    = results.monkeyRepr('ERROR', results.errors)
    
    # conditional adds here
    results = {
        many_modules_key(modules): from_namespace(locals(), RESULTS_TEMPLATE)
    }

    if options.subprocess:
        print TEST_RESULTS_START
        print pformat(results)
    else:
        return results

if __name__ == '__main__':
    options, args = opt_parser.parse_args()
    if not args: sys.exit('Called from run_tests.py, use that')
    run_test(args[0], options)
    
################################################################################