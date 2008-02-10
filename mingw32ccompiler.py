# module mingw32ccompiler.py
# Requires Python 2.1 or better.

"""Win32 GUI/console versions of the distutils mingw32 compiler classes."""

from distutils.cygwinccompiler import Mingw32CCompiler

def intersect (sequence_a, sequence_b):
    """Return true if the two sequences contain items in common

    If sequence_a is a non-sequence then return false.
    """
    try:
        for item in sequence_a:
            if item in sequence_b:
                return 1
    except TypeError:
        return 0
    return 0

def difference (sequence_a, sequence_b):
    """Return a list of items in sequence_a but not in sequence_b

    Will raise a ValueError if either argument is not a sequence.
    """
    new_sequence = []
    for item in sequence_a:
        if item not in sequence_b:
            new_sequence.append(item)
    return new_sequence

subsystem_options = ['-mwindows', '-mconsole']  # Item position is critical.

class Mingw32DefaultCCompiler (Mingw32CCompiler):
    """This mingw32 compiler class builds a Win32 GUI DLL by default.

    It is overridden by subsystem options in the linker extras.
    """

    def set_executables (self, **args):
        """Has no linker subsystem option for shared libraries"""
        Mingw32CCompiler.set_executables(self, **args)
        try:
            self.linker_so = difference (self.linker_so, subsystem_options)
        except TypeError:
            pass
                          
    def link (self,
              target_desc,
              objects,
              output_filename,
              output_dir=None,
              libraries=None,
              library_dirs=None,
              runtime_library_dirs=None,
              export_symbols=None,
              debug=0,
              extra_preargs=None,
              extra_postargs=None,
              build_temp=None,
              target_lang=None):
        """Do a Win32 GUI link if no subsystem option given."""

        if (target_desc != self.EXECUTABLE and
            not intersect(subsystem_options, extra_preargs) and
            not intersect(subsystem_options, extra_postargs)):
            try:
                extra_preargs = extra_preargs + subsystem_options[0]
            except TypeError:
                extra_preargs = subsystem_options[0:1]

        Mingw32CCompiler.link (self,
                               target_desc,
                               objects,
                               output_filename,
                               output_dir,
                               libraries,
                               library_dirs,
                               runtime_library_dirs,
                               export_symbols,
                               debug,
                               extra_preargs,
                               extra_postargs,
                               build_temp,
                               target_lang)

class Mingw32ConsoleCCompiler (Mingw32CCompiler):
    """This mingw32 compiler class builds a console DLL.

    It is not overridden by subsystem options in the linker extras.
    """

    def set_executables (self, **args):
        """Has console subsystem linker option for shared libraries."""
        Mingw32CCompiler.set_executables(self, **args)
        try:
            linker_so = difference(self.linker_so, subsystem_options)
        except TypeError:
            linker_so = subsystem_options[1:2]
        else:
            linker_so.append(subsystem_options[1])
        self.linker_so = linker_so
                          
    def link (self,
              target_desc,
              objects,
              output_filename,
              output_dir=None,
              libraries=None,
              library_dirs=None,
              runtime_library_dirs=None,
              export_symbols=None,
              debug=0,
              extra_preargs=None,
              extra_postargs=None,
              build_temp=None,
              target_lang=None):
        """Do a console link."""

        if target_desc != self.EXECUTABLE:
            try:
                extra_preargs = difference(extra_preargs, subsystem_options)
            except TypeError:
                pass
            try:
                extra_postargs = difference(extra_postargs, subsystem_options)
            except TypeError:
                pass
        Mingw32CCompiler.link (self,
                               target_desc,
                               objects,
                               output_filename,
                               output_dir,
                               libraries,
                               library_dirs,
                               runtime_library_dirs,
                               export_symbols,
                               debug,
                               extra_preargs,
                               extra_postargs,
                               build_temp,
                               target_lang)
