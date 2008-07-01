import subprocess, os, sys, re

DIFF_PROGRAM = r"H:\PortableApps\WinMergePortable\\WinMergePortable.exe"

IGNORE =  (
    '.svn',
    'infinite_loop',
)

NORMALIZERS = (
    (r"Ran (\d+) tests in (\d+\.\d+)s", "Ran XXX tests in X.XXXs" ),
    (r'File ".*?py",', 'File "XXXX.py",')  #TODO: look into why os.path.sep differs
)

main_dir  = os.path.split(os.path.abspath(sys.argv[0]))[0]
trunk_dir = os.path.normpath(os.path.join(main_dir, '../../'))

test_suite_dirs = [x for x in os.listdir(main_dir) if os.path.isdir(x)
                                                  and x not in IGNORE ]
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

def str_2_file(f, s):
    f = open(f, 'w')
    try:
        f.write(s)
    finally:
        f.close()

interactive = '-i' in sys.argv
if interactive: 
    print 'Running in interactive mode, will view diffs with %s' % DIFF_PROGRAM

single_cmd = "%s run_tests_sub.py -f %s"
subprocess_cmd = single_cmd + ' -s'

os.chdir(trunk_dir)
for suite in test_suite_dirs:
    single = call_proc(single_cmd % (sys.executable, suite))
    subs = call_proc(subprocess_cmd % (sys.executable, suite))

    normed_single = norm_result(single)
    normed_subs = norm_result(subs)

    if normed_single != normed_subs:
        if interactive:
            os.chdir(main_dir)
            
            single_file = "%s_single.txt" % suite
            subs_file = "%s_subs.txt" % suite
            
            str_2_file(single_file, normed_single)
            str_2_file(subs_file, normed_subs)
            
            subprocess.Popen([DIFF_PROGRAM, single_file, subs_file])
            sys.exit()
        else:
            raise Exception("%s suite output not matching" % suite)

print 'OK, output same'