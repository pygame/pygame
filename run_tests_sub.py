#!/usr/bin/env python

#################################### IMPORTS ###################################

import sys, os, re, unittest, subprocess, time

################################### CONSTANTS ##################################

IGNORE = (
	"scrap_test.py",
)

TEST_MODULE_RE = re.compile('^(.+_test\.py)$')

NUM_TESTS_RE = re.compile(r"Ran (\d+) tests?")

NUM_FAILS_RE = re.compile(r"failures=(\d+)")

NUM_ERRORS_RE = re.compile(r"errors=(\d+)")

DIV = "----------------------------------------------------------------------\nRan"

################################################################################

def proc_return_and_output(cmd, wd=None, env=None, bufsize=-1):
    if not isinstance(cmd, str) and sys.platform == "darwin":
        cmd = " ".join(cmd)

    proc = subprocess.Popen(
        cmd, cwd = wd, env = env, shell = True, bufsize = bufsize, 
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT
    )

    response = []
    finished = 0

    while not finished or proc.poll() is None:
        line = proc.stdout.readline()
        if not line: finished = 1
        else:
            response += [line.replace("\r\n", "\n").replace("\r", "\n")]

    return proc.wait(), ''.join(response)

################################################################################

def count_of(re, test_output):
    count = re.search(test_output)
    return count and int(count.group(1)) or 0

################################################################################

main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
test_subdir = os.path.join(main_dir, 'test')

sys.path.insert(0, test_subdir)
os.chdir( main_dir )

import test_utils

################################################################################

t = time.time()

total_tests = total_fails = total_errors = 0

for f in os.listdir(test_subdir):
    for module in TEST_MODULE_RE.findall(f):
        if module in IGNORE : continue

        ret_code, ret = proc_return_and_output('python test/%s' % module)
    
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