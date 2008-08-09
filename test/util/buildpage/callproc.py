################################################################################

import subprocess
import os
import sys

################################################################################

def get_cmd_str(cmd):
    if isinstance(cmd, str): 
        return cmd
    else:
        cmd = [c for c in cmd if c]
        if sys.platform == 'win32': cmd = subprocess.list2cmdline(cmd)
        else: cmd = ' '.join(cmd)
        return cmd

################################################################################

def ExecuteAssertSuccess(cmd, *args, **keywords):
    retcode, output = GetReturnCodeAndOutput(cmd, *args, **keywords)
    if retcode != 0:
        cmd_line = get_cmd_str(cmd)
        raise Exception("calling: "+cmd_line+" failed with output:\n"+output)
    return retcode, output

################################################################################

def GetReturnCodeAndOutput(cmd, dir=None, env=None, bufsize=-1, lineprintdiv=1):
    cmd = get_cmd_str(cmd)
    print "executing:", cmd, 'from dir', dir and dir or '.'
    
    proc = subprocess.Popen (
        cmd, cwd = dir, env = env, shell=True,
        bufsize = bufsize, 
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
        universal_newlines = 1,
    )
    
    response = []
    finished = False
    numlines = 0
    
    while not finished or proc.poll() == None:
        line = proc.stdout.readline()

        numlines += 1
        if numlines % lineprintdiv == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        response += [line]

        finished = line == ""

    sys.stdout.write("\n")
    return proc.wait(), ''.join(response)

################################################################################

def InteractiveGetReturnCodeAndOutput(cmd, input_string, dir=None, 
                                               env=None, bufsize=-1):
    cmd = get_cmd_str(cmd)
    print "executing:", cmd, 'from dir', dir and dir or '.'

    proc = subprocess.Popen (
        cmd, cwd=dir, env=env, shell=True, bufsize=bufsize,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines = 1
    )
        
    print "---------------"
    response = proc.communicate(input_string)[0]
    return proc.wait(), response

################################################################################