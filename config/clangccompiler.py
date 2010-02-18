"""A simple Intel C compiler class for Unix-based systems."""

from distutils.unixccompiler import UnixCCompiler
import distutils.sysconfig  as sysconfig

class ClangCCompiler(UnixCCompiler):
    """ClangCCompiler (verbose=0, dry_run=0, force=0) -> LLVMCCompiler
    
    A simple compiler class for the clang frontent of the LLVM Compiler
    infrastructure.
    """
    compiler_type = 'clang'

    def __init__ (self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__ (self, verbose, dry_run, force)
        cc = 'clang'

        cflags = sysconfig.get_config_var ('CFLAGS')
        cshared = sysconfig.get_config_var ('CCSHARED')
        ldflags = sysconfig.get_config_var ('LDFLAGS')

        self.set_executables(compiler=cc + ' ' + cflags,
                             compiler_so=cc + ' ' + cflags + ' ' + cshared,
                             compiler_cxx=cc,
                             linker_exe=cc + ' ' + ldflags,
                             linker_so=cc + ' -shared')
