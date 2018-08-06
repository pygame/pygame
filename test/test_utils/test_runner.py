import sys
import os

if __name__ == '__main__':
    pkg_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
    parent_dir, pkg_name = os.path.split(pkg_dir)
    is_pygame_pkg = (pkg_name == 'tests' and
                     os.path.split(parent_dir)[1] == 'pygame')
    if not is_pygame_pkg:
        sys.path.insert(0, parent_dir)
else:
    is_pygame_pkg = __name__.startswith('pygame.tests.')

import unittest
from .test_machinery import PygameTestLoader

import re
try:
    import StringIO
except ImportError:
    import io as StringIO

import optparse
from pprint import pformat


def prepare_test_env():
    test_subdir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
    main_dir = os.path.split(test_subdir)[0]
    sys.path.insert(0, test_subdir)
    fake_test_subdir = os.path.join(test_subdir, 'run_tests__tests')
    return main_dir, test_subdir, fake_test_subdir

main_dir, test_subdir, fake_test_subdir = prepare_test_env()

################################################################################
# Set the command line options
#
# options are shared with run_tests.py so make sure not to conflict
# in time more will be added here

TAG_PAT = r'-?[a-zA-Z0-9_]+'
TAG_RE = re.compile(TAG_PAT)
EXCLUDE_RE = re.compile("(%s,?\s*)+$" % (TAG_PAT,))

def exclude_callback(option, opt, value, parser):
    if EXCLUDE_RE.match(value) is None:
        raise optparse.OptionValueError("%s argument has invalid value" %
                                          (opt,))
    parser.values.exclude = TAG_RE.findall(value)

opt_parser = optparse.OptionParser()

opt_parser.add_option (
     "-i",  "--incomplete", action = 'store_true',
     help   = "fail incomplete tests" )

opt_parser.add_option (
     "-s",  "--usesubprocess", action = "store_true",
     help   = "run everything in a single process "
              " (default: use no subprocesses)" )

opt_parser.add_option (
     "-e",  "--exclude",
     action = 'callback',
     type   = 'string',
     help   = "exclude tests containing any of TAGS",
     callback = exclude_callback)

opt_parser.add_option (
     "-v",  "--unbuffered", action = 'store_true',
     help   = "Show stdout/stderr as tests run, rather than storing it and showing on failures" )

opt_parser.add_option (
     "-r",  "--randomize", action = 'store_true',
     help   = "randomize order of tests" )

################################################################################
# If an xxxx_test.py takes longer than TIME_OUT seconds it will be killed
# This is only the default, can be over-ridden on command line

TIME_OUT = 30

# DEFAULTS

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
return (first 10 and last 10 lines):
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
    unreliable.  See the dump option for reliable results.

    """

    all_dots = ''
    failures = []

    for module, results in sorted(all_results.items()):
        output, return_code, raw_return = map (
            results.get, ('output','return_code', 'raw_return')
        )

        if not output or (return_code and RAN_TESTS_DIV not in output):
            # would this effect the original dict? TODO
            output_lines = raw_return.splitlines()
            if len(output_lines) > 20:
                results['raw_return'] = '\n'.join(output_lines[:10] +
                                                  ['...'] +
                                                  output_lines[-10:]
                                                 )
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
TEST_RESULTS_END = "<--!! TEST RESULTS END HERE !!-->"
_test_re_str = '%s\n(.*)%s' % (TEST_RESULTS_START, TEST_RESULTS_END)
TEST_RESULTS_RE = re.compile(_test_re_str, re.DOTALL | re.M)

def get_test_results(raw_return):
    test_results = TEST_RESULTS_RE.search(raw_return)
    if test_results:
        try:
            return eval(test_results.group(1))
        except:
            print ("BUGGY TEST RESULTS EVAL:\n %s" % test_results.group(1))
            raise


################################################################################

def run_test(module, incomplete=False, usesubprocess=True, randomize=False,
             exclude=('interactive',), buffer=True):
    """Run a unit test module
    """
    suite = unittest.TestSuite()

    print ('loading %s' % module)

    loader = PygameTestLoader(randomize_tests=randomize,
                              include_incomplete=incomplete,
                              exclude=exclude)
    suite.addTest(loader.loadTestsFromName(module))

    output = StringIO.StringIO()
    runner = unittest.TextTestRunner(stream=output, buffer=buffer)
    results = runner.run(suite)

    results = {module: {
        'output': output.getvalue(),
        'num_tests': results.testsRun,
        'num_errors': len(results.errors),
        'num_failures': len(results.failures),
    }}

    if usesubprocess:
        print (TEST_RESULTS_START)
        print (pformat(results))
        print (TEST_RESULTS_END)
    else:
        return results

################################################################################

if __name__ == '__main__':
    options, args = opt_parser.parse_args()
    if not args:

        if is_pygame_pkg:
            run_from = 'pygame.tests.go'
        else:
            run_from = os.path.join(main_dir, 'run_tests.py')
        sys.exit('No test module provided; consider using %s instead' % run_from)
    run_test(args[0],
             incomplete=options.incomplete,
             usesubprocess=options.usesubprocess,
             randomize=options.randomize,
             exclude=options.exclude,
             buffer=(not options.unbuffered),
            )

################################################################################

