"""A simple Intel C compiler class for Unix-based systems."""

from distutils.unixccompiler import UnixCCompiler

class IntelCCompiler(UnixCCompiler):
    """IntelCCompiler (verbose=0, dry_run=0, force=0) -> IntelCCompiler
    
    A simple compiler class for Intel's C compiler.
    """
    compiler_type = 'intel'

    def __init__ (self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__ (self, verbose, dry_run, force)
        cc = 'icc'
        self.set_executables(compiler=cc,
                             compiler_so=cc,
                             compiler_cxx=cc,
                             linker_exe=cc,
                             linker_so=cc + ' -shared')
