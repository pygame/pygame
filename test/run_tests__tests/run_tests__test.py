import subprocess, os, sys, re

MERGE_PROGRAM = ""
INTERACTIVE = 0

IGNORE =  (
    '.svn',
    'infinite_loop',
)

NORMALIZERS = (
    ("Ran (\d+) tests in (\d+\.\d+)s", "Ran XXX tests in X.XXXs" ),
)

main_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
trunk_dir = os.path.normpath(os.path.join(main_dir, '../../'))

single_cmd = "%s run_tests_sub.py -f %s" 
subprocess_cmd = single_cmd + ' -s'

test_suite_dirs = [x for x in os.listdir(main_dir) if os.path.isdir(x)
                                                  and x not in IGNORE ]

def norm_result(result):
    """
    
    normalize differences, such as timing between output
    
    """

    for normalizer, replacement in NORMALIZERS:
        result = re.sub(normalizer, replacement, result)

    return result

def call_proc(cmd):
    proc = subprocess.Popen (
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell = 1,
    )
    assert not proc.wait()
    return proc.stdout.read()

os.chdir(trunk_dir)
for suite in test_suite_dirs:
    single = call_proc(single_cmd % (sys.executable, suite))[1]
    subs = call_proc(subprocess_cmd % (sys.executable, suite))[1]
    
    if norm_result(single) != norm_result(subs):
        if INTERACTIVE:
            pass ## open in MERGE_PROGRAM
        else:
            raise Exception("not matching")

print 'OK'