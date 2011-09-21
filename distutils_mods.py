#purpose: a few changes to distutils to build betterer.
#          - fixes up paths when using msys.
#
#

import os

import distutils.cygwinccompiler
distutils.cygwinccompiler.oldMingw32CCompiler= distutils.cygwinccompiler.Mingw32CCompiler
"""
Should put the above imports at the top of your file.
  and after them put

import distutils_mods
distutils.cygwinccompiler.Mingw32 = distutils_mods.mingcomp

"""

from distutils.errors import DistutilsExecError


class mingcomp(distutils.cygwinccompiler.oldMingw32CCompiler):

    def spawn(self, cmd):
        """  Because spawn uses a straight call to the systems underlying 
              shell, bypassing the string handling goodness of mingw/msys
              something gets fubared.  So this little hack method was put
              in its place.
        """
        self.verbose = 1
        if self.verbose:
            print "cmd :%s:" % cmd
        
        cmpl = " ".join(cmd)
        cmpl = cmpl.replace("\\", "/")
        cmpl = cmpl.replace("c:", "/c")
        cmpl = cmpl.replace("C:", "/c")
        cmpl = cmpl.replace("gcc", "gcc -g ")
        
        if self.verbose:
            print "cmpl is :%s:" % cmpl

        if not self.dry_run:
            import tempfile
            tmpfn = tempfile.mktemp(suffix='run_compiler')
            tmpf = open(tmpfn, "w+b")
            tmpf.write(cmpl)
            tmpf.close()
            r = os.system("sh %s" % tmpfn)

            os.remove(tmpfn)
            if r != 0:
                raise DistutilsExecError, \
                  "command '%s' failed with exit status :%d: command was :%s:.  " % (cmd[0], r, cmpl)
        
            if self.verbose:
                print "return value of the compile command is :%s:" % r




distutils.cygwinccompiler.Mingw32CCompiler= mingcomp








