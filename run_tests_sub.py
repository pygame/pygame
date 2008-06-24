#!/usr/bin/env python

#################################### IMPORTS ###################################

import sys, os, re, unittest, subprocess, time

main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
test_subdir = os.path.join(main_dir, 'test')
lib_subdir = os.path.join(main_dir, 'lib')

sys.path.insert(0, test_subdir)
sys.path.append(lib_subdir)

import test_utils, threadmap

os.chdir(main_dir)

################################### CONSTANTS ##################################

IGNORE = (
	"scrap_test.py",
)

TEST_MODULE_RE = re.compile('^(.+_test\.py)$')

NUM_TESTS_RE   = re.compile(r"Ran (\d+) tests?")
NUM_FAILS_RE   = re.compile(r"failures=(\d+)")
NUM_ERRORS_RE  = re.compile(r"errors=(\d+)")

DIV = (70 * "-") + "\nRan"

################################################################################

def test_return_and_output(cmd):
    test = cmd.split('/')[1]
    print 'Starting Test: %s' % test

    proc = subprocess.Popen (
        cmd, shell = True, bufsize = -1,
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT
    )

    ret_code = proc.wait()
    response = proc.stdout.read().replace("\r\n", "\n").replace("\r", "\n")

    return test, ret_code, response

################################################################################

def count_of(regex, test_output):
    count = regex.search(test_output)
    return count and int(count.group(1)) or 0

################################################################################

test_cmds = [('python test/%s' % f) for f in os.listdir(test_subdir) 
                                          if TEST_MODULE_RE.match(f) 
                                          and f not in IGNORE]
t = time.time()

if '-t' in sys.argv:
    tests = threadmap.tmap (
        test_return_and_output, test_cmds, 
        stop_on_error = False,
        num_workers = len(sys.argv) == 3 and int(sys.argv[2]) or 20
    )
else:
    tests = map(test_return_and_output, test_cmds)

################################################################################

total_tests = total_fails = total_errors = 0

complete_failure = {}

for module, ret_code, ret in tests:
    if ret_code:
        complete_failure[module] = ret_code, ret
        continue

    total_errors += count_of(NUM_ERRORS_RE, ret)
    total_fails  += count_of(NUM_FAILS_RE,  ret)
    total_tests  += count_of(NUM_TESTS_RE,  ret)
    
    print "%s %s" % (module, 'OK' in ret and 'OK' or ret.split(DIV)[0])

print "\n%s %s tests in %.3fs\n" % (DIV, total_tests, (time.time() - t))

if not total_errors and not total_fails:
    print 'OK'
else:
    print 'FAILED (failures=%s, errors=%s)' % (total_fails, total_errors)

################################################################################

if complete_failure: print '\n' + (70 * '=') +'\nComplete Failures\n'
for module, (ret_code, ret) in complete_failure.iteritems():
    print "%s failed with return code of %s" % (module, ret_code)
    print '\n' + ret

################################################################################