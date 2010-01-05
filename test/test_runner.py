################################################################################

try:
    import pygame2.test.pgunittest as unittest
except:
    import pgunittest as unittest

import sys, os, re, time, optparse
try:
    import StringIO as stringio
except:
    import io as stringio
from inspect import getdoc, getmembers, isclass
from pprint import pformat

try:
    import pygame2.test.unittest_patch as unittest_patch
    from pygame2.test.unittest_patch import StringIOContents
except:
    import unittest_patch
    from unittest_patch import StringIOContents

################################################################################

def prepare_test_env(directory):
    main_dir = os.path.split(directory)[0]
    test_subdir = main_dir #os.path.join (main_dir, "test")
    sys.path.insert(0, test_subdir)
    fake_test_subdir = os.path.join(test_subdir, 'run_tests__tests')
    return main_dir, test_subdir, fake_test_subdir

main_dir, test_subdir, fake_test_subdir = \
          prepare_test_env(os.path.abspath(__file__))

################################################################################
# Set the command line options
#
# options are shared with run_tests.py so make sure not to conflict
# in time more will be added here

opt_parser = optparse.OptionParser()

opt_parser.add_option (
     "-s",  "--subprocess", action = 'store_true',
     help   = "run everything in an own subprocess (default: single process)" )

opt_parser.add_option (
     "-d",  "--dump", action = 'store_true',
     help   = "dump failures/errors as dict ready to eval" )

opt_parser.add_option (
     "-F",  "--file",
     help   = "dump failures/errors to a file" )

opt_parser.add_option (
     "-T",  "--timings", type = 'int', default = 1, metavar = 'T',
     help   = "get timings for individual tests.\n" 
              "Run test T times, giving average time")

opt_parser.add_option (
     "-e",  "--exclude",
     help   = "exclude tests containing any of TAGS" )

opt_parser.add_option (
     "-w",  "--show_output", action = 'store_true',
     help   = "show silenced stderr/stdout on errors" )

opt_parser.add_option (
     "-a",  "--all", action = 'store_true',
     help   = "dump all results not just errors eg. -da" )

opt_parser.add_option (
     "-r",  "--randomize", action = 'store_true',
     help   = "randomize order of tests" )

opt_parser.add_option (
     "-S",  "--seed", type = 'int',
     help   = "seed randomizer" )

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

# If an xxxx_test.py takes longer than TIME_OUT seconds it will be killed
# This is only the default, can be over-ridden on command line

TIME_OUT = 30

# DEFAULTS

opt_parser.set_defaults (
    python = sys.executable,
    time_out = TIME_OUT,
    exclude = 'interactive',
)

################################################################################
# Human readable output
#

COMPLETE_FAILURE_TEMPLATE = """
======================================================================
ERROR: all_tests_for (%(module)s.AllTestCases)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test/%(module)s.py", line 1, in all_tests_for
subprocess completely failed with return code of %(return_code)s
cmd:          %(cmd)s
test_env:     %(test_env)s
working_dir:  %(working_dir)s
return (top 5 lines):
%(raw_return)s

"""  # Leave that last empty line else build page regex won't match
     # Text also needs to be vertically compressed
    

RAN_TESTS_DIV = (70 * "-") + "\nRan"

DOTS = re.compile("^([FE.]*)$", re.MULTILINE)

def combine_results(all_results, t):
    """

    Return pieced together results in a form fit for human consumption. Don't
    rely on results if  piecing together subprocessed  results (single process
    mode is fine). Was originally meant for that  purpose but was found to be
    unreliable.  See options.dump or options.human for reliable results.

    """

    all_dots = ''
    failures = []

    for module, results in sorted(all_results.items()):
        output, return_code, raw_return = map (
            results.get, ('output','return_code', 'raw_return')
        )

        if not output or (return_code and RAN_TESTS_DIV not in output):
            # would this effect the original dict? TODO
            results['raw_return'] = ''.join(raw_return.splitlines(1)[:5])
            failures.append( COMPLETE_FAILURE_TEMPLATE % results )
            all_dots += 'E'
            continue

        dots = DOTS.search(output).group(1)
        all_dots += dots

        if 'E' in dots or 'F' in dots:
            failures.append( output[len(dots)+1:].split(RAN_TESTS_DIV)[0] )
    
    total_fails, total_errors = map(all_dots.count, 'FE')
    total_tests = len(all_dots)

    combined = [all_dots]
    if failures: combined += [''.join(failures).lstrip('\n')[:-1]]
    combined += ["%s %s tests in %.3fs\n" % (RAN_TESTS_DIV, total_tests, t)]

    if not failures: combined += ['OK\n']
    else: combined += [
        'FAILED (%s)\n' % ', '.join (
            (total_fails  and ["failures=%s" % total_fails] or []) +
            (total_errors and ["errors=%s"  % total_errors] or [])
        )]

    return total_tests, '\n'.join(combined)

################################################################################

TEST_RESULTS_START = "<--!! TEST RESULTS START HERE !!-->"
TEST_RESULTS_RE = re.compile('%s\n(.*)' % TEST_RESULTS_START, re.DOTALL | re.M)

def get_test_results(raw_return):
    test_results = TEST_RESULTS_RE.search(raw_return)
    if test_results:
        try:     return eval(test_results.group(1))
        except:
            print ("BUGGY TEST RESULTS EVAL:\n %s" % test_results.group(1))
            raise

################################################################################
# ERRORS
# TODO

def make_complete_failure_error(result):
    return (
        "ERROR: all_tests_for (%s.AllTestCases)" % result['module'],
        "Complete Failure (ret code: %s)" % result['return_code'],
        result['test_file'], 
        '1',
    )
    
# For combined results, plural
def test_failures(results):
    errors = {}
    total =  sum(v.get('num_tests', 0) for v in results.values())
    for module, result in results.items():
        num_errors = (
            len(result.get('failures', [])) + len(result.get('errors', []))
        )
        if num_errors is 0 and result.get('return_code'):
            result.update(RESULTS_TEMPLATE)
            result['errors'].append(make_complete_failure_error(result))
            num_errors += 1
            total += 1
        if num_errors: errors.update({module:result})

    return total, errors

# def combined_errs(results):
#     for result in results.values():
#         combined_errs = result['errors'] + result['failures']
#         for err in combined_errs:
#             yield err

################################################################################
# For complete failures (+ namespace saving)

def from_namespace(ns, template):
    if isinstance(template, dict):
        return dict((i, ns.get(i, template[i])) for i in template)
    return dict((i, ns[i]) for i in template)

RESULTS_TEMPLATE = {
    'output'     :  '',
    'num_tests'  :   0,
    'failures'   :  [],
    'errors'     :  [],
    'tests'      :  {},
}

################################################################################

def run_test(module, options):
    suite = unittest.TestSuite()
    #test_utils.fail_incomplete_tests = options.incomplete

    m = __import__(module)

    if hasattr (m, "unittest") and m.unittest is not unittest:
        raise ImportError(
            "%s is not using correct unittest\n\n" % module +
            "should be: %s\n is using: %s" % (unittest.__file__,
                                              m.unittest.__file__)
        )
    
    print ("loading %s" % module)

    test = unittest.defaultTestLoader.loadTestsFromName(module)

    suite.addTest(test)

    output = stringio.StringIO()
    runner = unittest.TextTestRunner(stream = output)

    results = runner.run(suite)
    output  = StringIOContents(output)

    num_tests = results.testsRun
    failures  = results.failures
    errors    = results.errors
    tests     = results.tests

    results   = {module:from_namespace(locals(), RESULTS_TEMPLATE)}

    if options.subprocess:
        print (TEST_RESULTS_START)
        print (pformat (results))
    else:
        return (results)

################################################################################

if __name__ == '__main__':
    options, args = opt_parser.parse_args()
    unittest_patch.patch(options)
    if not args: sys.exit('Called from run_tests.py, use that')
    run_test(args[0], options)

################################################################################
