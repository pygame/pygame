import subprocess
import os
import sys

def ExecuteAssertSuccess(cmd, *args, **keywords):
    retcode, output = GetReturnCodeAndOutput(cmd, *args, **keywords)
    if retcode != 0:
        if isinstance(cmd, str):
            cmd_line = cmd
        else:
            cmd_line = " ".join(cmd)
        raise Exception("calling: "+cmd_line+" failed with output:\n"+output)
    return output

def GetReturnCodeAndOutput(cmd, dir=None, env=None, bufsize=-1, lineprintdiv=1):
    if isinstance(cmd, str):
        print "executing:",cmd
    else:           
        print "executing:"," ".join(cmd)
        if sys.platform == "darwin":
            cmd = " ".join(cmd)
    proc = subprocess.Popen(cmd, cwd=dir, env=env, shell=True, bufsize=bufsize, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    response = ""
    finished = False
    numlines = 0
    while not finished or proc.poll() == None:
        line = proc.stdout.readline()
        if line == "":
            finished = True
        else:
            numlines += 1
            if numlines % lineprintdiv == 0:
                sys.stdout.write(".")
            response += line.replace("\r\n", "\n").replace("\r", "\n")
    sys.stdout.write("\n")
    return proc.wait(), response

def InteractiveGetReturnCodeAndOutput(cmd, input_string, dir=None, env=None, bufsize=-1):
    if isinstance(cmd, str):
        print "executing:",cmd
    else:           
        print "executing:"," ".join(cmd)
        if sys.platform == "darwin":
            cmd = " ".join(cmd)
    proc = subprocess.Popen(cmd, cwd=dir, env=env, shell=True, bufsize=bufsize, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print "---------------"
    response = proc.communicate(input_string)[0]
    return proc.wait(), response
