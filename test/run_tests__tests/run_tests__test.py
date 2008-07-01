################################################################################

import subprocess, os, sys, re, difflib

################################################################################

IGNORE =  (
    '.svn',
    'infinite_loop',
)
NORMALIZERS = (
    (r"Ran (\d+) tests in (\d+\.\d+)s", "Ran XXX tests in X.XXXs" ),
    (r'File ".*?py",', 'File "XXXX.py",')  #TODO: look into why os.path.sep differs
)

################################################################################

def norm_result(result):
    "normalize differences, such as timing between output"
    for normalizer, replacement in NORMALIZERS:
        result = re.sub(normalizer, replacement, result)
    return result

def call_proc(cmd):
    proc = subprocess.Popen (
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell = 1,
    )
    assert not proc.wait()
    return proc.stdout.read()

################################################################################

main_dir  = os.path.split(os.path.abspath(sys.argv[0]))[0]
trunk_dir = os.path.normpath(os.path.join(main_dir, '../../'))

test_suite_dirs = [x for x in os.listdir(main_dir) if os.path.isdir(x)
                                                  and x not in IGNORE ]

################################################################################

single_cmd = "%s run_tests_sub.py -f %s"
subprocess_cmd = single_cmd + ' -s'

os.chdir(trunk_dir)
for suite in test_suite_dirs:
    single = call_proc(single_cmd % (sys.executable, suite))
    subs = call_proc(subprocess_cmd % (sys.executable, suite))

    normed_single, normed_subs = map(norm_result,(single, subs))

    if normed_single != normed_subs:
        print '%s suite FAILED\n' % suite
        print "difflib.Differ().compare(single, suprocessed):\n"
        print ''.join ( list(
            difflib.Differ().compare(
                normed_single.splitlines(1),
                normed_subs.splitlines(1)
            ))
        )
    else:
        print '%s suite OK' % suite

################################################################################