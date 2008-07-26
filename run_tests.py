#################################### IMPORTS ###################################
# TODO: clean up imports

import test.unittest as unittest

import sys, os, re, subprocess, time, optparse
import pygame.threads, pygame

from test_runner import prepare_test_env, run_test, combine_results, \
                        test_failures, get_test_results, from_namespace, \
                        TEST_RESULTS_START

from pprint import pformat

main_dir, test_subdir, fake_test_subdir = prepare_test_env()
test_runner_py = os.path.join(main_dir, "test_runner.py")

import test_utils, unittest_patch

################################### CONSTANTS ##################################
# Defaults:
#    See optparse options below for more options (test_runner.py)
#

# If an xxxx_test.py takes longer than TIME_OUT seconds it will be killed
# This is only the default, can be over-ridden on command line

TIME_OUT = 30

# Any tests in IGNORE will not be ran
IGNORE = set ([
    "scrap_test",
])

# Subprocess has less of a need to worry about interference between tests
SUBPROCESS_IGNORE = set ([
    "scrap_test",
])

INTERACTIVE = set ([
    'cdrom_test'
])

################################################################################
# Set the command line options
#
# Defined in test_runner.py as it shares options, added to here

from test_runner import opt_parser

opt_parser.set_usage("""

Runs all or some of the test/xxxx_test.py tests.

$ run_tests.py sprite threads -sd

Runs the sprite and threads module tests isolated in subprocesses, dumping all
failing tests info in the form of a dict.

""")

opt_parser.set_defaults (
    python = sys.executable,
    time_out = TIME_OUT,
)

options, args = opt_parser.parse_args()

################################################################################
# Change to working directory and compile a list of test modules
# If options.fake, then compile list of fake xxxx_test.py from run_tests__tests

TEST_MODULE_RE = re.compile('^(.+_test)\.py$')

if options.fake:
    test_subdir = os.path.join(fake_test_subdir, options.fake )
    sys.path.append(test_subdir)
    working_dir = test_subdir
else:
    working_dir = main_dir

test_env = {"PYTHONPATH": test_subdir}     #TODO:  append to PYTHONPATH
try:
    # Required by Python 2.6 on Windows.
    test_env["SystemRoot"] = os.environ["SystemRoot"]
except KeyError:
    pass
os.chdir(working_dir)

if args:
    test_modules = [
        m.endswith('_test') and m or ('%s_test' % m) for m in args
    ]
else:
    if options.subprocess: ignore = SUBPROCESS_IGNORE
    else: ignore = IGNORE

    # TODO: add option to run only INTERACTIVE, or include them, etc
    ignore = ignore | INTERACTIVE

    test_modules = []
    for f in sorted(os.listdir(test_subdir)):
        for match in TEST_MODULE_RE.findall(f):
            if match not in ignore:
                test_modules.append(match)

################################################################################
# Single process mode

if not options.subprocess:
    results = {}
    unittest_patch.patch(options)

    t = time.time()
    for module in test_modules:
        results.update(run_test(module, options = options))
    t = time.time() - t

################################################################################
# Subprocess mode
#

if options.subprocess:
    from async_sub import proc_in_time_or_kill

    def sub_test(module):
        print 'loading', module

        pass_on_args = [a for a in sys.argv[1:] if a not in args]
        cmd = [options.python, test_runner_py, module ] + pass_on_args

        return module, (cmd, test_env, working_dir), proc_in_time_or_kill (
            cmd, options.time_out,  env = test_env,  wd = working_dir,
        )

    if options.multi_thread:
        def tmap(f, args):
            return pygame.threads.tmap (
                f, args, stop_on_error = False,
                num_workers = options.multi_thread
            )
    else: tmap = map

    results = {}
    t = time.time()

    for module, cmd, (return_code, raw_return) in tmap(sub_test, test_modules):
        test_file = '%s.py' % os.path.join(test_subdir, module)
        cmd, test_env, working_dir = cmd

        test_results = get_test_results(raw_return)
        if test_results: results.update(test_results)
        else: results[module] = {}

        add_to_results = [
            'return_code', 'raw_return',  'cmd', 'test_file',
            'test_env', 'working_dir', 'module',
        ]

        results[module].update(from_namespace(locals(), add_to_results))
    
    t = time.time() -t

################################################################################
# Output Results
#

untrusty_total, combined = combine_results(results, t)
total, fails = test_failures(results)

if not options.subprocess: assert total == untrusty_total

if not options.dump or (options.human and untrusty_total == total):
    print combined
else:
    print TEST_RESULTS_START
    print pformat(options.all and results or fails)

################################################################################