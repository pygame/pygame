#!/usr/bin/env python

"""

Test runner for pygame unittests:

By default, runs all test/xxxx_test.py files in a single process.

Option to run tests in subprocesses using subprocess and async_sub. Will poll 
tests for return code and if tests don't return after TIME_OUT, will kill 
process with os.kill. On win32 platform win32api.TerminateProcess is used.

Dependencies:
    async_sub.py:
        Requires win32 extensions when run on windows

"""

#################################### IMPORTS ###################################

import sys, os, re, unittest, subprocess, time, optparse
import pygame.threads 

# async_sub imported if needed when run in subprocess mode

main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
test_subdir = os.path.join(main_dir, 'test')
fake_test_subdir = os.path.join(test_subdir, 'run_tests__tests')

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

subprocess completely failed with return code of %(ret_code)s

cmd: %(cmd)s

return (abbrv):
%(ret)s

"""  # Leave that last empty line else build page regex won't match

RAN_TESTS_DIV = (70 * "-") + "\nRan"

DOTS = re.compile("^([FE.]*)$", re.MULTILINE)

TEST_MODULE_RE = re.compile('^(.+_test)\.py$')

################################################################################
# Set the command line options
#

USEAGE = """

Runs all the test/xxxx_test.py tests.

"""

opt_parser = optparse.OptionParser(USEAGE)
opt_parser.add_option(
     "-v",  "--verbose", action = 'store_true',
     help   = "be verbose in output (only single process mode)" )

opt_parser.add_option (
     "-i",  "--incomplete", action = 'store_true',
     help   = "fail incomplete tests (only single process mode)" )

opt_parser.add_option (
     "-s",  "--subprocess", action = 'store_true',
     help   = "run test suites in subprocesses (default: same process)" )

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

os.chdir(main_dir)

test_modules = []
for f in sorted(os.listdir(test_subdir)):
    for match in TEST_MODULE_RE.findall(f):
        test_modules.append(match)

################################################################################
# Run all the tests in one process
# unittest.TextTestRunner().run(unittest.TestSuite())
#

if not options.subprocess:
    suite = unittest.TestSuite()
    runner = unittest.TextTestRunner()
        
    for module in [m for m in test_modules if m not in IGNORE]:
        print 'loading ' + module
        __import__( module )
        test = unittest.defaultTestLoader.loadTestsFromName( module )
        suite.addTest( test )
    
    test_utils.fail_incomplete_tests = options.incomplete
    if options.verbose:
        runner.verbosity = 2
    
    runner.run( suite )
    
    sys.exit()

    ###########################
    # SYS.EXIT() FLOW CONTROL #
    ###########################

################################################################################
# Runs an individual xxxx_test.py test suite in a subprocess
#

import async_sub

def run_test(args):
    module, cmd = args
    print 'loading %s' % module
    ret_code, response = async_sub.proc_in_time_or_kill (
        cmd, time_out = options.time_out
    )
    return cmd, module, ret_code, response

################################################################################
# Run all the tests in subprocesses
#

flags = []
test_cmds = [ 
    (m, [options.python, os.path.join(test_subdir, '%s.py' % m)] + flags)
        for m in test_modules if  m not in SUBPROCESS_IGNORE 
]


t = time.time()

if options.multi_thread:
    test_results = pygame.threads.tmap (
        run_test, test_cmds,
        stop_on_error = False,
        num_workers = options.multi_thread
    )
else:
    test_results = map(run_test, test_cmds)

t = time.time() - t

################################################################################
# Combine subprocessed TextTestRunner() results to mimick single run
# Puts complete failures in a form the build page will pick up

all_dots = ''
failures = []

for cmd, module, ret_code, ret in test_results:
    if ret_code and RAN_TESTS_DIV not in ret:
        ret = ''.join(ret.splitlines(1)[:5])
        failures.append( COMPLETE_FAILURE_TEMPLATE % locals() )
        continue

    dots = DOTS.search(ret).group(1)
    all_dots += dots

    if 'E' in dots or 'F' in dots:
        failure = ret[len(dots):].split(RAN_TESTS_DIV)[0]
        failures.append (
            failure.replace( "(__main__.", "(%s." % module)
        )

total_fails, total_errors = map(all_dots.count, 'FE')
total_tests = len(all_dots)

print all_dots
if failures: print ''.join(failures).lstrip('\n')[:-1]
print "%s %s tests in %.3fs\n" % (RAN_TESTS_DIV, total_tests, t)

if not failures:
    print 'OK'
else:
    print 'FAILED (%s)' % ', '.join (
        (total_fails  and ["failures=%s" % total_fails] or []) +
        (total_errors and ["errors=%s"  % total_errors] or [])
    )

################################################################################