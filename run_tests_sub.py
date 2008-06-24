#!/usr/bin/env python

#################################### IMPORTS ###################################

import sys, os, re, unittest, subprocess, time, threadmap

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

def proc_return_and_output(cmd):
    proc = subprocess.Popen (
        cmd, shell = True, bufsize=-1,
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT
    )

    ret_code = proc.wait()
    response = proc.stdout.read().replace("\r\n", "\n").replace("\r", "\n")

    return ret_code, response

################################################################################

def count_of(regex, test_output):
    count = regex.search(test_output)
    return count and int(count.group(1)) or 0

################################################################################

main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
test_subdir = os.path.join(main_dir, 'test')

sys.path.insert(0, test_subdir)
os.chdir(main_dir)

import test_utils

################################################################################

t = time.time()

total_tests = total_fails = total_errors = 0

tests = [('python test/%s' % f) for f in os.listdir(test_subdir) 
                                      if TEST_MODULE_RE.match(f)]

tests = threadmap.tmap(proc_return_and_output, tests, stop_on_error = False)

print time.time() -t

# for f in os.listdir(test_subdir):
#     for module in TEST_MODULE_RE.findall(f):
#         if module in IGNORE : continue

#         ret_code, ret = proc_return_and_output('python test/%s' % module)
    
#         total_errors += count_of(NUM_ERRORS_RE, ret)
#         total_fails  += count_of(NUM_FAILS_RE,  ret)
#         total_tests  += count_of(NUM_TESTS_RE,  ret)

#         print "%s %s" % (module, 'OK' in ret and 'OK' or ret.split(DIV)[0])
        
# print "\n%s %s tests in %.3fs\n" % (DIV, total_tests, (time.time() - t))

# if not total_errors and not total_fails:
#     print 'OK'
# else:
#     print 'FAILED (failures=%s, errors=%s)' % (total_fails, total_errors)

################################################################################