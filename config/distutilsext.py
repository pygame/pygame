"""Enhance distutils by various compilers."""

from distutils import ccompiler
from distutils.errors import DistutilsModuleError

# Add the compiler classes to the ccompiler table. Unfortunate hacks
# follow.
compiler_class = ccompiler.compiler_class
value = compiler_class['mingw32']
assert len(value) == 3, "distutils.ccompiler.compiler_class has changed"

compiler_class['mingw32'] = ("mingw32ccompiler", "MinGW32CCompiler",
                             value[2] + ", Win32 GUI shared libraries defaullt")
compiler_class['mingw32-console'] = ("mingw32ccompiler", "MinGW32CCompiler",
                                     value[2] + ", console shared libraries")
compiler_class['icc'] = ("intelccompiler", "IntelCCompiler", "Intel C Compiler")
compiler_class['clang'] = ("clangccompiler", "ClangCCompiler",
                           "Clang LLVM C Compiler")

original_new_compiler = ccompiler.new_compiler
def new_compiler (plat=None,
                  compiler=None,
                  verbose=0,
                  dry_run=0,
                  force=0):
    """Recognizes replacement mingw32 compiler classes"""

    if compiler == 'mingw32':
        from config.mingw32ccompiler import Mingw32DefaultCCompiler
        return Mingw32DefaultCCompiler (None, dry_run, force)
    if compiler == 'mingw32-console':
        from config.mingw32ccompiler import Mingw32ConsoleCCompiler
        return Mingw32ConsoleCCompiler (None, dry_run, force)
    if compiler == 'icc':
        from config.intelccompiler import IntelCCompiler
        return IntelCCompiler (None, dry_run, force)
    if compiler == 'clang':
        from config.clangccompiler import ClangCCompiler
        return ClangCCompiler (None, dry_run, force)
    return original_new_compiler (plat, compiler, verbose, dry_run, force)

ccompiler.new_compiler = new_compiler
