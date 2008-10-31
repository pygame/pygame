################################################################################

# StdLib
import subprocess
import os
import sys

# User Libs
from helpers import ProgressIndicator

################################################################################

def get_cmd_str(cmd):
    if isinstance(cmd, str): 
        return cmd
    else:
        cmd = [c for c in cmd if c]
        if sys.platform == 'win32': cmd = subprocess.list2cmdline(cmd)
        else: cmd = ' '.join(cmd)
        return cmd

def log_cmd(cmd, dir):
    print "executing:", cmd, 'from dir', os.path.abspath(dir or os.getcwd())

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
    log_cmd(cmd, dir)
    
    proc = subprocess.Popen (
        cmd, cwd = dir, env = env, shell = True, bufsize = bufsize, 
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
        universal_newlines = 1,
    )

    response = []
    progress = ProgressIndicator(lineprintdiv)

    while proc.poll() is None:
        response += [proc.stdout.readline()]
        if response[-1] is "": break
        progress()

    progress.finish()
    return proc.wait(), ''.join(response) + proc.stdout.read() # needed ubuntu

################################################################################

def InteractiveGetReturnCodeAndOutput(cmd, input_string, dir=None, 
                                               env=None, bufsize=-1):
    cmd = get_cmd_str(cmd)
    log_cmd(cmd, dir)

    proc = subprocess.Popen (
        cmd, cwd = dir, env = env, shell = True, bufsize = bufsize, 
        stdin = subprocess.PIPE,    stdout = subprocess.PIPE, 
        stderr = subprocess.STDOUT, universal_newlines = 1
    )
        
    print "---------------"
    response = proc.communicate(input_string)[0]
    return proc.wait(), response

################################################################################