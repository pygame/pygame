#################################### IMPORTS ##################################

if __name__ == '__main__':
    import sys
    sys.exit("This module is for import only")

test_pkg_name = '.'.join(__name__.split('.')[0:-2])
is_pygame_pkg = test_pkg_name == 'pygame.tests'
if is_pygame_pkg:
    from pygame.tests import test_utils, IGNORE, SUBPROCESS_IGNORE
    from pygame.tests.test_utils import unittest, unittest_patch
    from pygame.tests.test_utils.test_runner \
         import prepare_test_env, run_test, combine_results, test_failures, \
                get_test_results, from_namespace, TEST_RESULTS_START, \
                opt_parser
else:
    from test import test_utils, IGNORE, SUBPROCESS_IGNORE
    from test.test_utils import unittest, unittest_patch
    from test.test_utils.test_runner \
         import prepare_test_env, run_test, combine_results, test_failures, \
                get_test_results, from_namespace, TEST_RESULTS_START, \
                opt_parser
import pygame
import pygame.threads

import sys
import os
import re
import subprocess
import time
import optparse
import random
from pprint import pformat

was_run = False

def run(*args, **kwds):
    """Run the Pygame unit test suite and return (total tests run, fails dict)

    Positional arguments (optional):
    The names of tests to include. If omitted then all tests are run. Test names
    need not include the trailing '_test'.

    Keyword arguments:
    incomplete - fail incomplete tests (default False)
    subprocess - run test suites in subprocesses (default False, same process)
    dump - dump failures/errors as dict ready to eval (default False)
    file - if provided, the name of a file into which to dump failures/errors
    timings - if provided, the number of times to run each individual test to
              get an average run time (default is run each test once)
    exclude - A list of TAG names to exclude from the run
    show_output - show silenced stderr/stdout on errors (default False)
    all - dump all results, not just errors (default False)
    randomize - randomize order of tests (default False)
    seed - if provided, a seed randomizer integer
    multi_thread - if provided, the number of THREADS in which to run
                   subprocessed tests
    time_out - if subprocess is True then the time limit in seconds before
               killing a test (default 30)
    fake - if provided, the name of the fake tests package in the
           run_tests__tests subpackage to run instead of the normal
           Pygame tests
    python - the path to a python executable to run subprocessed tests
             (default sys.executable)

    Return value:
    A tuple of total number of tests run, dictionary of error information.
    The dictionary is empty if no errors were recorded.

    Tests can be run in the current process or a subprocess, depending on
    the 'subprocess' argument. If tests are run in separate processes then
    frozen tests will be killed. Also, using subprocesses more realistically
    reproduces normal Pygame usage, where pygame.init() and pygame.quit() are
    called only once per program execution.

    Tests are run in a randomized order if the randomize argument is True
    or a seed argument is provided. If no seed integer is provided then
    the system time is used.

    Individual test modules may have a __tags__ attribute, a list of tag strings
    used to selectively omit modules from a run. By default only 'interactive'
    modules such as cdrom_test are ignored. An interactive module must be run
    from the console as a Python program.

    This function can only be called once per Python session. It is not
    reentrant.

    """

    global was_run

    if was_run:
        raise RuntimeError("run() was already called this session")
    was_run = True
                           
    options = kwds.copy()
    option_subprocess = options.get('subprocess', False)
    option_dump = options.pop('dump', False)
    option_file = options.pop('file', None)
    option_all = options.pop('all', False)
    option_randomize = options.get('randomize', False)
    option_seed = options.get('seed', None)
    option_multi_thread = options.pop('multi_thread', 1)
    option_time_out = options.pop('time_out', 30)
    option_fake = options.pop('fake', None)
    option_python = options.pop('python', sys.executable)

    main_dir, test_subdir, fake_test_subdir = prepare_test_env()
    test_runner_py = os.path.join(test_subdir, "test_utils", "test_runner.py")
    cur_working_dir = os.path.abspath(os.getcwd())

    ###########################################################################
    # Compile a list of test modules. If fake, then compile list of fake
    # xxxx_test.py from run_tests__tests

    TEST_MODULE_RE = re.compile('^(.+_test)\.py$')

    test_mods_pkg_name = test_pkg_name
    
    if option_fake is not None:
        test_mods_pkg_name = '.'.join([test_mods_pkg_name,
                                       'run_tests__tests',
                                       option_fake])
        test_subdir = os.path.join(fake_test_subdir, option_fake)
        working_dir = test_subdir
    else:
        working_dir = main_dir


    # Added in because some machines will need os.environ else there will be
    # false failures in subprocess mode. Same issue as python2.6. Needs some
    # env vars.

    test_env = os.environ

    fmt1 = '%s.%%s' % test_mods_pkg_name
    fmt2 = '%s.%%s_test' % test_mods_pkg_name
    if args:
        test_modules = [
            m.endswith('_test') and (fmt1 % m) or (fmt2 % m) for m in args
        ]
    else:
        if option_subprocess:
            ignore = SUBPROCESS_IGNORE
        else:
            ignore = IGNORE

        test_modules = []
        for f in sorted(os.listdir(test_subdir)):
            for match in TEST_MODULE_RE.findall(f):
                if match not in ignore:
                    test_modules.append(fmt1 % match)

    ###########################################################################
    # Meta results

    results = {}
    meta_results = {'__meta__' : {}}
    meta = meta_results['__meta__']

    ###########################################################################
    # Randomization

    if option_randomize or option_seed is not None:
        if option_seed is None:
            option_seed = time.time()
        meta['random_seed'] = option_seed
        print "\nRANDOM SEED USED: %s\n" % option_seed
        random.seed(option_seed)
        random.shuffle(test_modules)

    ###########################################################################
    # Single process mode

    if not option_subprocess:
        unittest_patch.patch(**kwds)

        t = time.time()
        for module in test_modules:
            results.update(run_test(module, **kwds))
        t = time.time() - t

    ###########################################################################
    # Subprocess mode
    #

    if option_subprocess:
        if is_pygame_pkg:
            from pygame.tests.test_utils.async_sub import proc_in_time_or_kill
        else:
            from test.test_utils.async_sub import proc_in_time_or_kill

        pass_on_args = []
        for option in ['timings', 'exclude', 'seed']:
            value = options.pop(option, None)
            if value is not None:
                pass_on_args.append('--%s' % option)
                pass_on_args.append(str(value))
        for option, value in options.items():
            if value:
                pass_on_args.append('--%s' % option)

        def sub_test(module):
            print 'loading', module

            cmd = [option_python, test_runner_py, module ] + pass_on_args

            return (module,
                    (cmd, test_env, working_dir),
                    proc_in_time_or_kill(cmd, option_time_out, env=test_env,
                                         wd=working_dir))

        if option_multi_thread > 1:
            def tmap(f, args):
                return pygame.threads.tmap (
                    f, args, stop_on_error = False,
                    num_workers = option_multi_thread
                )
        else:
            tmap = map

        t = time.time()

        for module, cmd, (return_code, raw_return) in tmap(sub_test,
                                                           test_modules):
            test_file = '%s.py' % os.path.join(test_subdir, module)
            cmd, test_env, working_dir = cmd

            test_results = get_test_results(raw_return)
            if test_results:
                results.update(test_results)
            else:
                results[module] = {}

            add_to_results = [
                'return_code', 'raw_return',  'cmd', 'test_file',
                'test_env', 'working_dir', 'module',
            ]

            results[module].update(from_namespace(locals(), add_to_results))

        t = time.time() - t

    ###########################################################################
    # Output Results
    #

    untrusty_total, combined = combine_results(results, t)
    total, fails = test_failures(results)

    meta['total_tests'] = total
    meta['combined'] = combined
    results.update(meta_results)

    if not option_subprocess:
        assert total == untrusty_total

    if not option_dump:
        print combined
    else:
        results = option_all and results or fails
        print TEST_RESULTS_START
        print pformat(results)

    if option_file is not None:
        results_file = open(option_file, 'w')
        try:
            results_file.write(pformat(results))
        finally:
            results_file.close()

    return total, fails

###############################################################################

