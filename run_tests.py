#################################### IMPORTS ###################################

import sys, os, re, unittest, subprocess, time, optparse
import pygame.threads 

from test_runner import run_test, TEST_RESULTS_RE
from pprint import pformat

# async_sub imported if needed when run in subprocess mode

main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
test_subdir = os.path.join(main_dir, 'test')
fake_test_subdir = os.path.join(test_subdir, 'run_tests__tests')
test_runner_py = os.path.join(main_dir, "test_runner.py")

sys.path.insert(0, test_subdir)
import test_utils

################################### CONSTANTS ##################################
# Defaults:
#    See optparse options below for more options
#

# If an xxxx_test.py takes longer than TIME_OUT seconds it will be killed
# This is only the default, can be over-ridden on command line

TIME_OUT = 30

# Any tests in IGNORE will not be ran
IGNORE = (
    "scrap_test",
)

# Subprocess has less of a need to worry about interference between tests
SUBPROCESS_IGNORE = (
    "scrap_test",
)

################################################################################

COMPLETE_FAILURE_TEMPLATE = """
======================================================================
ERROR: all_tests_for (%(module)s.AllTestCases)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test\%(module)s.py", line 1, in all_tests_for

subprocess completely failed with return code of %(return_code)s

cmd:          %(cmd)s
test_env:     %(test_env)s
working_dir:  %(working_dir)s

return (top 5 lines):
%(raw_return)s

"""  # Leave that last empty line else build page regex won't match

TEST_MODULE_RE = re.compile('^(.+_test)\.py$')

RAN_TESTS_DIV = (70 * "-") + "\nRan"

DOTS = re.compile("^([FE.]*)$", re.MULTILINE)

################################################################################
# Set the command line options
#

USEAGE = """

Runs all the test/xxxx_test.py tests.

"""

opt_parser = optparse.OptionParser(USEAGE)

opt_parser.add_option (
     "-i",  "--incomplete", action = 'store_true',
     help   = "fail incomplete tests (only single process mode)" )

opt_parser.add_option (
     "-s",  "--subprocess", action = 'store_true',
     help   = "run test suites in subprocesses (default: same process)" )

opt_parser.add_option (
     "-d",  "--dump", action = 'store_true',
     help   = "dump results as dict ready to eval" )

opt_parser.add_option (
     "-m",  "--multi_thread", metavar = 'THREADS', type = 'int',
     help   = "run subprocessed tests in x THREADS" )

opt_parser.add_option (
     "-t",  "--time_out", metavar = 'SECONDS', type = 'int', default = TIME_OUT,
     help   = "kill stalled subprocessed tests after SECONDS" )

opt_parser.add_option (
     "-f",  "--fake", metavar = "DIR",
     help   = "run fake tests in %s%s$DIR"  % (fake_test_subdir, os.path.sep) )

opt_parser.add_option (
     "-p",  "--python", metavar = "PYTHON", default = sys.executable,
     help   = "path to python excutable to run subproccesed tests\n"
              "default (sys.executable): %s" % sys.executable)

options, args = opt_parser.parse_args()

################################################################################
# Change to working directory and compile a list of test modules
# If options.fake, then compile list of fake xxxx_test.py from run_tests__tests
# this is used for testing subprocess output against single process mode

if options.fake:
    test_subdir = os.path.join(fake_test_subdir, options.fake )
    sys.path.append(test_subdir)
    working_dir = test_subdir
else:
    working_dir = main_dir

test_env = {"PYTHONPATH": test_subdir}
os.chdir(working_dir)

test_modules = []
for f in sorted(os.listdir(test_subdir)):
    for match in TEST_MODULE_RE.findall(f):
        test_modules.append(match)

################################################################################
# Single process mode
#

if not options.subprocess:
    test_utils.fail_incomplete_tests = options.incomplete
    single_results = run_test([m for m in test_modules if m not in IGNORE])
    if options.dump: print pformat(single_results)
    else: print single_results['output']

################################################################################
# Subprocess mode
#

def count(results, *args):
    for arg in args:
        all_of = [a for a in [v.get(arg) for v in results.values()] if a]
        if not all_of: yield 0
        else:
            yield sum (
            isinstance(all_of[0], int) and all_of or (len(v) for v in all_of)
        )

def combine_results(all_results, t):
    """

    Return pieced together subprocessed results in a form fit for human 
    consumption. Don't rely on results. Was originally meant for that purpose 
    but was found to be unreliable. See options.dump for reliable results.

    """

    all_dots = ''
    failures = []

    for module, results in sorted(all_results.items()):
        output, return_code, raw_return = map (
            results.get, ('output','return_code', 'raw_return')
        )

        if not output or (return_code and RAN_TESTS_DIV not in output):
            results['raw_return'] = ''.join(raw_return.splitlines(1)[:5])
            failures.append( COMPLETE_FAILURE_TEMPLATE % results )
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

if options.subprocess:
    from async_sub import proc_in_time_or_kill

    def sub_test(module):
        print 'loading', module
        
        cmd = [options.python, test_runner_py, module ]

        return module, (cmd, test_env, working_dir), proc_in_time_or_kill (
            cmd,
            options.time_out,
            env = test_env,
            wd = working_dir,
        )
    
    if options.multi_thread:
        def tmap(f, args):
            return pygame.threads.tmap (
                f, args, stop_on_error = False,
                num_workers = options.multi_thread
            )
    else: tmap = map
        

    test_modules = (m for m in test_modules if m not in SUBPROCESS_IGNORE)
    results = {}

    t = time.time()

    for module, proc, (return_code, raw_return) in tmap(sub_test, test_modules):
        cmd, test_env, working_dir = proc

        test_results = TEST_RESULTS_RE.search(raw_return)
        if test_results: 
            try:     results.update(eval(test_results.group(1)))
            except:  raise Exception("BUGGY EVAL:\n %s" % test_results.group(1))

        else: results[module] = {}

        results[module].update (
            {
                'return_code': return_code,
                'raw_return' : raw_return,
                'cmd'        : cmd,
                'test_env'   : test_env,
                'working_dir': working_dir,
                'module'     : module,
            }
        )

    untrusty_total, combined = combine_results(results, time.time() -t)
    errors, failures, total  = count(results, 'errors', 'failures', 'num_tests')

    if not options.dump and untrusty_total == total:
        print combined
    else:
        for module, result in results.items():
            for breaker in ['errors', 'return_code', 'failures']:
                if breaker not in result or result[breaker]:
                    print pformat(result)

        print "Tests:%s Errors:%s Failures:%s"% (total, errors, failures)

################################################################################