import sys, os, re, unittest, StringIO, time, optparse
from pprint import pformat

################################################################################

def prepare_test_env():
    main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
    test_subdir = os.path.join(main_dir, 'test')
    sys.path.insert(0, test_subdir)
    return main_dir, test_subdir

main_dir, test_subdir = prepare_test_env()
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
     "-a",  "--all", action = 'store_true',
     help   = "dump all results not just errors eg. -da" )

opt_parser.add_option (
     "-H",  "--human", action = 'store_true',
     help   = "dump results as dict ready to eval if unsure" 
              " (subprocess mode)" ) # TODO

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

def redirect_output():
    yield sys.stderr, sys.stdout
    sys.stderr, sys.stdout = StringIO.StringIO(), StringIO.StringIO()
    yield sys.stderr, sys.stdout

def restore_output(err, out):
    sys.stderr, sys.stdout = err, out

def StringIOContents(io):
    io.seek(0)
    return io.read()
    
unittest._TextTestResult.monkeyRepr = lambda self, errors: [
    (self.getDescription(e[0]), e[1]) for e in errors
]

def run_test(modules, options):
    if isinstance(modules, str): modules = [modules]
    suite = unittest.TestSuite()

    if not options.fake:
        import test_utils
        test_utils.fail_incomplete_tests = options.incomplete

    for module in modules:
        __import__(module)
        print 'loading', module

        # filter test by tags based on options
        test = unittest.defaultTestLoader.loadTestsFromName(module)
        suite.addTest(test)

    (realerr, realout), (err, out) =  redirect_output()
    # restore_output(realerr, realout)   #DEBUG

    captured = StringIO.StringIO()
    runner = unittest.TextTestRunner(stream = captured)
    results = runner.run(suite)

    captured, err, out = map(StringIOContents, (captured, err, out))
    restore_output(realerr, realout)

    results = (
        {
            options.subprocess and modules[0] or 'all_tests':
            {
                'num_tests' : results.testsRun,
                'failures'  : results.monkeyRepr(results.failures),
                'errors'    : results.monkeyRepr(results.errors),
                'output'    : captured,
                'stderr'    : err,
                'stdout'    : out,
            }
        }
    )

    if options.subprocess:
        print TEST_RESULTS_START
        print pformat(results)
    else:
        return results['all_tests']

if __name__ == '__main__':
    options, args = opt_parser.parse_args()
    if not args: sys.exit('Called from run_tests.py, use that')
    run_test(args[0], options)