#################################### IMPORTS ###################################
import sys
import os
import re
import subprocess
import time
import optparse
import pygame2.threads, pygame2
import random

try:
    from pygame2.test.test_runner import prepare_test_env, run_test, \
         combine_results, test_failures, get_test_results, from_namespace, \
         TEST_RESULTS_START
except:
    from test_runner import prepare_test_env, run_test, combine_results, \
         test_failures, get_test_results, from_namespace, \
         TEST_RESULTS_START

from pprint import pformat

main_dir, test_subdir, fake_test_subdir = \
          prepare_test_env(os.path.abspath(__file__))
test_runner_py = os.path.join(main_dir, "test_runner.py")

try:
    import pygame2.test.unittest_patch as unittest_patch
except:
    import unittest_patch as unittest_patch

def run ():
    global test_subdir
    global main_dir
    global fake_test_subdir
    ############################# CONSTANTS ##############################
    # Defaults:
    #    See optparse options below for more options (test_runner.py)
    #
    #######################################################################
    # Set the command line options
    #
    # Defined in test_runner.py as it shares options, added to here
    try:
        from pygame2.test.test_runner import opt_parser
    except:
        from test_runner import opt_parser

    opt_parser.set_usage("""

    Runs all or some of the test/xxxx_test.py tests.

    $ run_tests.py sprite threads -sd

    Runs the sprite and threads module tests isolated in subprocesses,
    dumping all failing tests info in the form of a dict.

    """)

    options, args = opt_parser.parse_args()

    ########################################################################
    # Change to working directory and compile a list of test modules If
    # options.fake, then compile list of fake xxxx_test.py from
    # run_tests__tests

    TEST_MODULE_RE = re.compile('^(.+_test)\.py$')

    if options.fake:
        test_subdir = os.path.join(fake_test_subdir, options.fake )
        sys.path.append(test_subdir)
        working_dir = test_subdir
    else:
        working_dir = main_dir

    # Added in because some machines will need os.environ else there
    # will be false failures in subprocess mode. Same issue as
    # python2.6. Needs some env vars.
    test_env = os.environ.copy()
    test_env["PYTHONPATH"] = os.pathsep.join (
        [pth for pth in ([test_subdir] + [test_env.get("PYTHONPATH")]) if pth]
        )

    os.chdir(working_dir)

    if args:
        test_modules = [
            m.endswith('_test') and m or ('%s_test' % m) for m in args
            ]
    else:

        test_modules = []
        for f in sorted(os.listdir(test_subdir)):
            for match in TEST_MODULE_RE.findall(f):
                test_modules.append(match)

    #######################################################################
    # Meta results

    results = {}
    meta_results = {'__meta__' : {}}
    meta = meta_results['__meta__']

    #######################################################################
    # Randomization

    if options.randomize or options.seed:
        seed = options.seed or time.time()
        meta['random_seed'] = seed
        print ("\nRANDOM SEED USED: %s\n" % seed)
        random.seed(seed)
        random.shuffle(test_modules)
        
    #######################################################################
    # Single process mode

    if not options.subprocess:
        unittest_patch.patch(options)

        t = time.time()
        for module in test_modules:
            results.update(run_test(module, options = options))
        t = time.time() - t

    ######################################################################
    # Subprocess mode
    #

    if options.subprocess:
        from async_sub import proc_in_time_or_kill

        def sub_test(module):
            print ('loading %s' % module)

            pass_on_args = [a for a in sys.argv[1:] if a not in args]
            cmd = [options.python, test_runner_py, module ] + pass_on_args

            return module, (cmd, test_env, working_dir), proc_in_time_or_kill (
                cmd, options.time_out,  env = test_env,  wd = working_dir,
                )

        if options.multi_thread:
            def tmap(f, args):
                return pygame2.threads.tmap (
                    f, args, stop_on_error = False,
                    num_workers = options.multi_thread
                    )
        else: tmap = map

        t = time.time()

        for module, cmd, (return_code, raw_return) in \
                tmap(sub_test, test_modules):
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

    ######################################################################
    # Output Results
    #

    untrusty_total, combined = combine_results(results, t)
    total, fails = test_failures(results)

    meta['total_tests'] = total
    meta['combined'] = combined
    results.update(meta_results)

    if not options.subprocess:
        assert total == untrusty_total

    if not options.dump:
        print (combined)
    else:
        results = options.all and results or fails
        print (TEST_RESULTS_START)
        print (pformat(results))

    if options.file:
        results_file = open(options.file, 'w')
        try:        results_file.write(pformat(results))
        finally:    results_file.close()

if __name__ == "__main__":
    run ()
