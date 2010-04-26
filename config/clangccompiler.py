"""A simple LLVM C compiler class for Unix-based systems."""
import os

from distutils.unixccompiler import UnixCCompiler
import distutils.sysconfig  as sysconfig

class ClangCCompiler(UnixCCompiler):
    """ClangCCompiler (verbose=0, dry_run=0, force=0) -> ClangCCompiler
    
    A simple compiler class for the clang frontend of the LLVM Compiler
    infrastructure.
    """
    compiler_type = 'clang'

    def __init__ (self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__ (self, verbose, dry_run, force)
        cc = 'clang'

        cflags = sysconfig.get_config_var ('CFLAGS') or ""
        cflags += os.getenv('CFLAGS', '')
        cshared = sysconfig.get_config_var ('CCSHARED') or ""
        ldflags = sysconfig.get_config_var ('LDFLAGS') or ""
        ldflags += os.getenv('LDFLAGS', '')
        cppflags = os.getenv ('CPPFLAGS', '')

        cargs = ' ' + cflags + ' ' + cppflags
        soargs = ' ' + cflags + ' ' + cshared
        ldargs = ' ' + ldflags
        
        self.set_executables(compiler=cc + cargs,
                             compiler_so=cc + soargs,
                             compiler_cxx=cc,
                             linker_exe=cc + ldflags,
                             linker_so=cc + ' -shared')
