#################################### IMPORTS ###################################

import sys, os, re, unittest, subprocess, time, optparse
import pygame.threads 

from test_runner import run_test, get_test_results, TEST_RESULTS_START,\
                        prepare_test_env, from_namespace, many_modules_key,\
                        count, test_failures

from pprint import pformat

main_dir, test_subdir, fake_test_subdir = prepare_test_env()
test_runner_py = os.path.join(main_dir, "test_runner.py")

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
# Human readable output
#

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
# this is used for testing subprocess output against single process mode

if options.fake:
    test_subdir = os.path.join(fake_test_subdir, options.fake )
    sys.path.append(test_subdir)
    working_dir = test_subdir
else:
    working_dir = main_dir

test_env = {"PYTHONPATH": test_subdir}
os.chdir(working_dir)

if args:
    test_modules = [
        m.endswith('_test') and m or ('%s_test' % m) for m in args
    ]
else:
    if options.subprocess: ignore = SUBPROCESS_IGNORE
    else: ignore = IGNORE

    test_modules = []
    for f in sorted(os.listdir(test_subdir)):
        for match in TEST_MODULE_RE.findall(f):
            if match in ignore: continue
            test_modules.append(match)

################################################################################
# Single process mode
#

if not options.subprocess:
    single_results = run_test ( test_modules, options = options)
    if options.dump: print pformat(single_results)
    #TODO  make consistent with subprocess mode
    else: print single_results[many_modules_key(test_modules)]['output']

################################################################################
# Subprocess mode
#

def combine_results(all_results, t):
    """

    Return pieced together subprocessed results in a form fit for human 
    consumption. Don't rely on results. Was originally meant for that purpose 
    but was found to be unreliable. 
    
    See options.dump or options.human for reliable results.

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
        # conditional adds here

        results[module].update(from_namespace(locals(), add_to_results))

    untrusty_total, combined = combine_results(results, time.time() -t)
    total, fails = test_failures(results)

    if not options.dump or (options.human and untrusty_total == total):
        print combined
    else:
        print TEST_RESULTS_START
        # print pformat(list(combined_errs(fails)))
        print pformat(options.all and results or fails)

################################################################################