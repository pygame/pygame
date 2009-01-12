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

main_dir, test_subdir, fake_test_subdir = prepare_test_env()
test_runner_py = os.path.join(test_subdir, "test_utils", "test_runner.py")
cur_working_dir = os.path.abspath(os.getcwd())

def run(caller_name=None):
    global test_subdir

    if  caller_name is None:
        caller_name = ('python -c "import %(pkg)s; %(pkg)s.run()"' %
                       {'pkg': test_pkg_name})
        
    ###########################################################################
    # Set the command line options
    #
    # Defined in test_runner.py as it shares options, added to here

    opt_parser.set_usage("""

    Runs all or some of the %(pkg)s.xxxx_test tests.

    $ %(exec)s sprite threads -sd

    Runs the sprite and threads module tests isolated in subprocesses, dumping
    all failing tests info in the form of a dict.

    """ % {'pkg': test_pkg_name, 'exec': caller_name})

    options, args = opt_parser.parse_args()

    ###########################################################################
    # Change to working directory and compile a list of test modules
    # If options.fake, then compile list of fake xxxx_test.py from
    # run_tests__tests

    TEST_MODULE_RE = re.compile('^(.+_test)\.py$')

    test_mods_pkg_name = test_pkg_name
    
    if options.fake:
        test_mods_pkg_name = '.'.join([test_mods_pkg_name,
                                       'run_tests__tests',
                                       options.fake])
        test_subdir = os.path.join(fake_test_subdir, options.fake )
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
        if options.subprocess:
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

    if options.randomize or options.seed:
        seed = options.seed or time.time()
        meta['random_seed'] = seed
        print "\nRANDOM SEED USED: %s\n" % seed
        random.seed(seed)
        random.shuffle(test_modules)

    ###########################################################################
    # Single process mode

    if not options.subprocess:
        unittest_patch.patch(options)

        t = time.time()
        for module in test_modules:
            results.update(run_test(module, options = options))
        t = time.time() - t

    ###########################################################################
    # Subprocess mode
    #

    if options.subprocess:
        if is_pygame_pkg:
            from pygame.tests.test_utils.async_sub import proc_in_time_or_kill
        else:
            from test.test_utils.async_sub import proc_in_time_or_kill

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

    if not options.subprocess:
        assert total == untrusty_total

    if not options.dump:
        print combined
    else:
        results = options.all and results or fails
        print TEST_RESULTS_START
        print pformat(results)

    if options.file:
        results_file = open(options.file, 'w')
        try:
            results_file.write(pformat(results))
        finally:
            results_file.close()


###############################################################################

