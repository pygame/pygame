import sys, os, re, unittest, StringIO, time
from pprint import pformat

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
    
unittest._TextTestResult.monkey = lambda self, errors: [ 
    (self.getDescription(e[0]), e[1]) for e in errors
]

def run_test(module, sub_process_mode=False):
    suite = unittest.TestSuite()
    if not isinstance(module, list): module = [module]

    for modules in module:   
        __import__(modules)
        print 'loading', modules
        test = unittest.defaultTestLoader.loadTestsFromName(modules)
        suite.addTest(test)
    
    (realerr, realout), (err, out) =  redirect_output()
    # restore_output(realerr, realout)   DEBUG
    
    captured = StringIO.StringIO()
    runner = unittest.TextTestRunner(stream = captured)
    results = runner.run( suite )

    captured, err, out = map(StringIOContents, (captured, err, out))
    restore_output(realerr, realout)

    results = (
        {
            len(module) == 1 and module[0] or 'all_tests':
            {
                'num_tests' : results.testsRun,
                'failures'  : results.monkey(results.failures),
                'errors'    : results.monkey(results.errors),
                'output'    : captured,
                'stderr'    : err,
                'stdout'    : out,
            }
        }
    )

    if sub_process_mode:
        print TEST_RESULTS_START
        print pformat(results)
    else:
        return results['all_tests']

if __name__ == '__main__':
    run_test(sys.argv[1], 1)