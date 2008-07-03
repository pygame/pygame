################################################################################

import subprocess, os, sys, re, difflib

################################################################################

IGNORE =  (
    '.svn',
    'infinite_loop',
)
NORMALIZERS = (
    (r"Ran (\d+) tests in (\d+\.\d+)s",   "Ran \\1 tests in X.XXXs" ),
    (r'File ".*?([^/\\.]+\.py)"',         'File "\\1"')  
    #TODO: look into why os.path.sep differs
)

################################################################################

def norm_result(result):
    "normalize differences, such as timing between output"
    for normalizer, replacement in NORMALIZERS:
        if callable(normalizer):
            result = normalizer(result)
        else:
            result = re.sub(normalizer, replacement, result)
    
    return result

def call_proc(cmd):
    proc = subprocess.Popen (
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell = 1,
    )
    assert not proc.wait()
    return proc.stdout.read()

################################################################################

unnormed_diff = '-u' in sys.argv
verbose = '-v' in sys.argv or unnormed_diff
if '-h' in sys.argv or '--help' in sys.argv: sys.exit (
    '-v, to output diffs even on success\n'
    '-u, to output diffs of unnormalized tests'
)

main_dir  = os.path.split(os.path.abspath(sys.argv[0]))[0]
trunk_dir = os.path.normpath(os.path.join(main_dir, '../../'))

test_suite_dirs = [x for x in os.listdir(main_dir) 
                           if os.path.isdir(os.path.join(main_dir, x))
                           and x not in IGNORE ]

################################################################################
# Test that output is the same in single process and subprocess modes 
#

single_cmd = "%s run_tests.py -f %s"
subprocess_cmd = single_cmd + ' -s'

passes = 0
failed = False

os.chdir(trunk_dir)
for suite in test_suite_dirs:
    single = call_proc(single_cmd % (sys.executable, suite))
    subs = call_proc(subprocess_cmd % (sys.executable, suite))

    normed_single, normed_subs = map(norm_result,(single, subs))

    failed = normed_single != normed_subs
    if failed:
        print '%s suite FAILED\n' % suite    
    else:
        passes += 1
        print '%s suite OK' % suite
    
    if verbose or failed:
        print "difflib.Differ().compare(single, suprocessed):\n"
        print ''.join ( list(
            difflib.Differ().compare (
                (unnormed_diff and single or normed_single).splitlines(1),
                (unnormed_diff and subs or normed_subs).splitlines(1)
            ))
        )

print "\n%s/%s passes" % (passes, len(test_suite_dirs))
print "\n-h for help"

################################################################################