################################################################################

import subprocess, os, sys, re, difflib

################################################################################

IGNORE =  (
    '.svn',
    'infinite_loop',
)
NORMALIZERS = (
    (r"Ran (\d+) tests in (\d+\.\d+)s",   "Ran \\1 tests in X.XXXs" ),
    (r'File ".*?([^/\\.]+\.py)"',         'File "\\1"'),
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

def call_proc(cmd, cd=None):
    proc = subprocess.Popen (
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd = cd,
	universal_newlines = True,
    )
    if proc.wait():
        print cmd, proc.wait()
        raise Exception(proc.stdout.read())

    return proc.stdout.read()

################################################################################

unnormed_diff = '-u' in sys.argv
verbose = '-v' in sys.argv or unnormed_diff
if '-h' in sys.argv or '--help' in sys.argv: sys.exit (
    "\nCOMPARES OUTPUT OF SINGLE VS SUBPROCESS MODE OF RUN_TESTS.PY\n\n"
    '-v, to output diffs even on success\n'
    '-u, to output diffs of unnormalized tests\n\n'
    "Each line of a Differ delta begins with a two-letter code:\n\n"
    "    '- '    line unique to sequence 1\n"
    "    '+ '    line unique to sequence 2\n"
    "    '  '    line common to both sequences\n"
    "    '? '    line not present in either input sequence\n"
)

main_dir  = os.path.split(os.path.abspath(sys.argv[0]))[0]
trunk_dir = os.path.normpath(os.path.join(main_dir, '../../'))

test_suite_dirs = [x for x in os.listdir(main_dir) 
                           if os.path.isdir(os.path.join(main_dir, x))
                           and x not in IGNORE ]

################################################################################
# Test that output is the same in single process and subprocess modes 
#

base_cmd = [sys.executable, 'run_tests.py']

cmd = base_cmd + ['-f']
sub_cmd = base_cmd + ['-s', '-f']
time_out_cmd =  base_cmd  + ['-t', '4', '-s', '-f', 'infinite_loop' ]

passes = 0
failed = False

for suite in test_suite_dirs:
    single = call_proc(cmd + [suite], trunk_dir)
    subs = call_proc(sub_cmd + [suite], trunk_dir)

    normed_single, normed_subs = map(norm_result,(single, subs))

    failed = normed_single != normed_subs
    if failed:
        print '%s suite comparison FAILED\n' % suite    
    else:
        passes += 1
        print '%s suite comparison OK' % suite
    
    if verbose or failed:
        print "difflib.Differ().compare(single, suprocessed):\n"
        print ''.join ( list(
            difflib.Differ().compare (
                (unnormed_diff and single or normed_single).splitlines(1),
                (unnormed_diff and subs or normed_subs).splitlines(1)
            ))
        )

print "infinite_loop suite (subprocess mode timeout)",
loop_test = call_proc(time_out_cmd, trunk_dir)
assert "successfully terminated" in loop_test
passes += 1
print "OK"

print "\n%s/%s passes" % (passes, len(test_suite_dirs) + 1)

print "\n-h for help"

################################################################################