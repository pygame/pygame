#!/usr/bin/env python
# -*- coding: ascii -*-
# Program msys_link_VC_2008_dlls.py
# Requires Python 2.4 or later and win32api.

"""Link dependency DLLs against the Visual C 2008 run-time using MinGW and MSYS

Configured for Pygame 1.8 and Python 2.6 and up.

By default the DLLs and export libraries are installed in directory ./lib_VC_2008.
msys_build_deps.py must run first to build the static libaries.

This program can be run from a Windows cmd.exe or MSYS terminal.

The recognized, and optional, environment variables are:
  SHELL - MSYS shell program path - already defined in the MSYS terminal
  LDFLAGS - linker options - prepended to flags set by the program
  LIBRARY_PATH - library directory paths - appended to those used by this
                 program

To get a list of command line options run

python build_deps.py --help

This program has been tested against the following libraries:

SDL 1.2 (.13) revision 4114 from SVN 
SDL_image 1.2.6
SDL_mixer 1.2 (.8) revision 3942 from SVN
SDL_ttf 2.0.9
smpeg revision 370 from SVN
freetype 2.3.7
libogg 1.1.3
libvorbis 1.2.0
FLAC 1.2.1
tiff 3.8.2
libpng 1.2.32
jpeg 6b
zlib 1.2.3

The build environment used:

gcc-core-3.4.5
binutils-2.17.50
mingwrt-3.15.1
win32api-3.12
pexports 0.43
MSYS-1.0.10

Builds have been performed on Windows 98 and XP.

Build issues:
  For pre-2007 computers:  MSYS bug "[ 1170716 ] executing a shell scripts
    gives a memory leak" (http://sourceforge.net/tracker/
    index.php?func=detail&aid=1170716&group_id=2435&atid=102435)
    
    It may not be possible to use the --all option to build all Pygame
    dependencies in one session. Instead the job may need to be split into two
    or more sessions, with a reboot of the operatingsystem between each. Use
    the --help-args option to list the libraries in the their proper build
    order.
"""

import msys

from optparse import OptionParser, Option, OptionValueError
import os
import sys
import time
import re
import copy

DEFAULT_DEST_DIR_NAME = 'lib_VC_2008'

def print_(*args, **kwds):
    msys.msys_print(*args, **kwds)

def merge_strings(*args, **kwds):
    """Returns non empty string joined by sep

    The default separator is an empty string.
    """

    sep = kwds.get('sep', '')
    return sep.join([s for s in args if s])

class BuildError(StandardError):
    """Raised for missing source paths and failed script runs"""
    pass

class Dependency(object):
    """Builds a library"""
    
    def __init__(self, name, dlls, shell_script):
        self.name = name
        self.dlls = dlls
        self.shell_script = shell_script

    def build(self, msys):
        return_code = msys.run_shell_script(self.shell_script)
        if return_code != 0:
            raise BuildError("The build for %s failed with code %d" %
                             (self.name, return_code))

class Preparation(object):
    """Perform necessary build environment preperations"""
    
    def __init__(self, name, shell_script):
        self.name = name
        self.path = ''
        self.paths = []
        self.dlls = []
        self.shell_script = shell_script

    def build(self, msys):
        return_code = msys.run_shell_script(self.shell_script)
        if return_code != 0:
            raise BuildError("Preparation '%s' failed with code %d" %
                             (self.name, return_code))

def build(dependencies, msys):
    """Execute that shell scripts for all dependencies"""
    
    for dep in dependencies:
        dep.build(msys)

def check_directory_path(option, opt, value):
    # Remove those double quotes that Windows won't.
    if re.match(r'([A-Za-z]:){0,1}[^"<>:|?*]+$', value) is None:
        raise OptionValueError("option %s: invalid path" % value)
    return value

class MyOption(Option):
    TYPES = Option.TYPES + ("dir",)
    TYPE_CHECKER = copy.copy(Option.TYPE_CHECKER)
    TYPE_CHECKER["dir"] = check_directory_path

def command_line():
    """Process the command line and return the options"""
    
    usage = ("usage: %prog [options] --all\n"
             "       %prog [options] [args]\n"
             "\n"
             "Build the Pygame dependencies. The args, if given, are\n"
             "libraries to include or exclude.\n"
             "\n"
             "At startup this program may prompt for missing information.\n"
             "Be aware of this before redirecting output or leaving the\n"
             "program unattended. Once the 'Starting build' message appears\n"
             "no more user input is required. The build process will"
             "abort on the first error, as library build order is important.\n"
             "\n"
             "See --include and --help-args.\n"
             "\n"
             "For more details see the program's document string\n")
    
    parser = OptionParser(usage, option_class=MyOption)
    parser.add_option('-a', '--all', action='store_true', dest='build_all',
                      help="Include all libraries in the build")
    parser.set_defaults(build_all=False)
    parser.add_option('--console', action='store_true', dest='console',
                      help="Link with the console subsystem:"
                           " defaults to Win32 GUI")
    parser.set_defaults(console=False)
    parser.add_option('--no-strip', action='store_false', dest='strip',
                      help="Do not strip the library")
    parser.set_defaults(strip=True)
    parser.add_option('-e', '--exclude', action='store_true', dest='exclude',
                      help="Exclude the specified libraries")
    parser.set_defaults(exclude=False)
    parser.add_option('-d', '--destination-dir', type='dir',
                      dest='destination_dir',
                      help="Where the DLLs and export libraries will go",
                      metavar='PATH')
    parser.set_defaults(destination_dir=DEFAULT_DEST_DIR_NAME)
    parser.add_option('-m', '--msys-root', action='store', type='dir',
                      dest='msys_directory',
                      help="MSYS directory path, which may include"
                           " the 1.x subdirectory")
    parser.add_option('--help-args', action='store_true', dest='arg_help',
                      help="Show a list of recognised libraries,"
                           " in build order, and exit")
    parser.set_defaults(arg_help=False)
    return parser.parse_args()

def set_environment_variables(msys, options):
    """Set the environment variables used by the scripts"""
    
    environ = msys.environ
    msys_root = msys.msys_root
    destination_dir = os.path.abspath(options.destination_dir)
    environ['BDWD'] = msys.windows_to_msys(destination_dir)
    environ['BDBIN'] = '/usr/local/bin'
    environ['BDLIB'] = '/usr/local/lib'
    subsystem = '-mwindows'
    if options.console:
        subsystem = '-mconsole'
    strip = ''
    if options.strip:
        strip = '-Wl,--strip-all'
    environ['LDFLAGS'] = merge_strings(environ.get('LDFLAGS', ''),
                                       subsystem,
                                       strip,
                                       sep=' ')
    library_path = os.path.join(msys_root, 'local', 'lib')
    msvcr90_path = os.path.join(destination_dir, 'msvcr90')
    environ['DBMSVCR90'] = msys.windows_to_msys(msvcr90_path)
    # For dependency libraries and msvcrt hiding.
    environ['LIBRARY_PATH'] = merge_strings(msvcr90_path,
                                            environ.get('LIBRARY_PATH', ''),
                                            sep=';')

class ChooseError(StandardError):
    """Failer to select dependencies"""
    pass

def choose_dependencies(dependencies, options, args):
    """Return the dependencies to actually build"""

    if options.build_all:
        if args:
            raise ChooseError("No library names are accepted"
                              " for the --all option.")
        if options.exclude:
            return []
        else:
            return dependencies

    if args:
        names = [d.name for d in dependencies]
        args = [a.upper() for a in args]
        for a in args:
            if a not in names:
                msg = ["%s is an unknown library; valid choices are:" % a]
                msg.extend(names)
                raise ChooseError('\n'.join(msg))
        if options.exclude:
            return [d for d in dependencies if d.name not in args]
        return [d for d in dependencies if d.name in args]

    return []
    
def summary(dependencies, msys, start_time, chosen_deps, options):
    """Display a summary report of new, existing and missing DLLs"""

    import datetime

    print_("\n\n=== Summary ===")
    if start_time is not None:
        print_("  Elapse time:",
               datetime.timedelta(seconds=time.time()-start_time))
    bin_dir = options.destination_dir
    for d in dependencies:
        name = d.name
        dlls = d.dlls
        for dll in dlls:
            dll_path = os.path.join(bin_dir, dll)
            try:
                mod_time = os.path.getmtime(dll_path)
            except:
                msg = "No DLL"
            else:
                if mod_time >= start_time:
                    msg = "Installed new DLL %s" % dll_path
                else:
                    msg = "-- (old DLL %s)" % dll_path
            print_("  %-10s: %s" % (name, msg))
    
def main(dependencies, msvcr90_preparation, msys_preparation):
    """Build the dependencies according to the command line options."""

    options, args = command_line()
    if options.arg_help:
        print_("These are the Pygame library dependencies:")
        for dep in dependencies:
            print_(" ", dep.name)
        return 0
    try:
        chosen_deps = choose_dependencies(dependencies, options, args)
    except ChooseError, e:
        print_(e)
        return 1
    print_("Destination directory:", options.destination_dir)
    if not chosen_deps:
        if not args:
            print_("No libraries specified.")
        elif options.build_all:
            print_("All libraries excluded")
    chosen_deps.insert(0, msvcr90_preparation)
    chosen_deps.insert(0, msys_preparation)
    try:
        msys_directory = options.msys_directory
    except AttributeError:
        msys_directory = None
    try:
        m = msys.Msys(msys_directory)
    except msys.MsysException, e:
        print_(e)
        return 1
    start_time = None
    return_code = 1
    set_environment_variables(m, options)
    print_("\n=== Starting build ===")
    start_time = time.time()  # For file timestamp checks.
    try:
        build(chosen_deps, m)
    except BuildError, e:
        print_("Build aborted:", e)
    else:
        # A successful build!
        return_code = 0
    summary(dependencies, m, start_time, chosen_deps, options)

    return return_code

#
#   Build specific code
#

# This list includes the MSYS shell scripts to build each library. Each script
# runs in an environment where MINGW_ROOT_DIRECTORY is defined and the MinGW
# bin directory is in PATH. DBWD, is the working directory. A script will cd to
# it before doing anything else. BDBIN is the location of the dependency DLLs.
# BDLIB is the location of the dependency libraries. LDFLAGS are linker flags.
# 
# The list order corresponds to build order. It is critical.
dependencies = [
    Dependency('SDL', ['SDL.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/SDL.dll" >SDL.def
gcc -shared $LDFLAGS -o SDL.dll -def SDL.def "$BDLIB/libSDL.a" -lwinmm -ldxguid
dlltool -D SDL.dll -d SDL.def -l libSDL.dll.a
ranlib libSDL.dll.a
strip --strip-all SDL.dll
"""),
    Dependency('Z', ['zlib1.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/zlib1.dll" >z.def
gcc -shared $LDFLAGS -o zlib1.dll -def z.def "$BDLIB/libz.a"
dlltool -D zlib1.dll -d z.def -l libz.dll.a
ranlib libz.dll.a
strip --strip-all zlib1.dll
"""),
    Dependency('FREETYPE', ['libfreetype-6.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/libfreetype-6.dll" >freetype.def
gcc -shared $LDFLAGS -L. -o libfreetype-6.dll -def freetype.def \
  "$BDLIB/libfreetype.a" -lz
dlltool -D libfreetype-6.dll -d freetype.def -l libfreetype.dll.a
ranlib libfreetype.dll.a
strip --strip-all libfreetype-6.dll
"""),
    Dependency('FONT', ['SDL_ttf.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/SDL_ttf.dll" >SDL_ttf.def
gcc -shared $LDFLAGS -L. "-L$BDLIB" -o SDL_ttf.dll -def SDL_ttf.def \
  "$BDLIB/libSDL_ttf.a" -lSDL -lfreetype
dlltool -D SDL_ttf.dll -d SDL_ttf.def -l libSDL_ttf.dll.a
ranlib libSDL_ttf.dll.a
strip --strip-all SDL_ttf.dll
"""),
    Dependency('PNG', ['libpng12-0.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/libpng12-0.dll" >png.def
gcc -shared $LDFLAGS -L. -o libpng12-0.dll -def png.def "$BDLIB/libpng.a" -lz
dlltool -D libpng12-0.dll -d png.def -l libpng.dll.a
ranlib libpng.dll.a
strip --strip-all libpng12-0.dll
"""),
    Dependency('JPEG', ['jpeg.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/jpeg.dll" >jpeg.def
gcc -shared $LDFLAGS -o jpeg.dll -def jpeg.def "$BDLIB/libjpeg.a"
dlltool -D jpeg.dll -d jpeg.def -l libjpeg.dll.a
ranlib libjpeg.dll.a
strip --strip-all jpeg.dll
"""),
    Dependency('TIFF', ['libtiff.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/libtiff.dll" >tiff.def
gcc -shared $LDFLAGS -L. -o libtiff.dll -def tiff.def \
  "$BDLIB/libtiff.a" -ljpeg -lz
dlltool -D libtiff.dll -d tiff.def -l libtiff.dll.a
ranlib libtiff.dll.a
strip --strip-all libtiff.dll
"""),
    Dependency('IMAGE', ['SDL_image.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/SDL_image.dll" >SDL_image.def
gcc -shared $LDFLAGS -L. -o SDL_image.dll -def SDL_image.def \
  "$BDLIB/libSDL_image.a" -lSDL -ljpeg -lpng -ltiff
dlltool -D SDL_image.dll -d SDL_image.def -l libSDL_image.dll.a
ranlib libSDL_image.dll.a
strip --strip-all SDL_image.dll
"""),
    Dependency('SMPEG', ['smpeg.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/smpeg.dll" >smpeg.def
g++ -shared $LDFLAGS -L. -o smpeg.dll -def smpeg.def \
  "$BDLIB/libsmpeg.a" -lSDL
dlltool -D smpeg.dll -d smpeg.def -l libsmpeg.dll.a
ranlib libsmpeg.dll.a
strip --strip-all smpeg.dll
"""),
    Dependency('OGG', ['libogg-0.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/libogg-0.dll" >ogg.def
gcc -shared $LDFLAGS -o libogg-0.dll -def ogg.def "$BDLIB/libogg.a"
dlltool -D libogg-0.dll -d ogg.def -l libogg.dll.a
ranlib libogg.dll.a
strip --strip-all libogg-0.dll
"""),
    Dependency('VORBIS', ['libvorbis-0.dll', 'libvorbisfile-3.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/libvorbis-0.dll" >vorbis.def
gcc -shared $LDFLAGS -L. -o libvorbis-0.dll -def vorbis.def \
  "$BDLIB/libvorbis.a" -logg
dlltool -D libvorbis-0.dll -d vorbis.def -l libvorbis.dll.a
ranlib libvorbis.dll.a
strip --strip-all libvorbis-0.dll

pexports "$BDBIN/libvorbisfile-3.dll" >vorbisfile.def
gcc -shared $LDFLAGS -L. -o libvorbisfile-3.dll -def vorbisfile.def \
  "$BDLIB/libvorbisfile.a" -lvorbis -logg
dlltool -D libvorbisfile-3.dll -d vorbisfile.def -l libvorbisfile.dll.a
ranlib libvorbisfile.dll.a
strip --strip-all libvorbisfile-3.dll
"""),
    Dependency('MIXER', ['SDL_mixer.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/SDL_mixer.dll" >SDL_mixer.def
gcc -shared $LDFLAGS -L. -L/usr/local/lib -o SDL_mixer.dll -def SDL_mixer.def \
  "$BDLIB/libSDL_mixer.a" -lSDL -lsmpeg -lvorbisfile -lFLAC -lWs2_32 -lwinmm
dlltool -D SDL_mixer.dll -d SDL_mixer.def -l libSDL_mixer.dll.a
ranlib libSDL_mixer.dll.a
strip --strip-all SDL_mixer.dll
"""),
    Dependency('PORTMIDI', ['portmidi.dll'], """

set -e
cd "$BDWD"

pexports "$BDBIN/portmidi.dll" >portmidi.def
gcc -shared $LDFLAGS -L. -L/usr/local/lib -o portmidi.dll -def portmidi.def \
  "$BDLIB/libportmidi.a" -lwinmm
dlltool -D portmidi.dll -d portmidi.def -l portmidi.dll.a
ranlib libSDL_mixer.dll.a
strip --strip-all portmidi.dll
"""),
    ]  # End dependencies = [.


msys_prep = Preparation('/usr/local', """

# Ensure destination directories exists.
mkdir -p "$BDWD"
mkdir -p "$DBMSVCR90"
""")
    
msvcr90_prep = Preparation('msvcr90.dll linkage', r"""

set -e

#
#   msvcr90.dll support
#
if [ ! -f "$DBMSVCR90/libmoldnamed.dll.a" ]; then
  OBJS='isascii.o iscsym.o iscsymf.o toascii.o
        strcasecmp.o strncasecmp.o wcscmpi.o'
  if [ ! -d /tmp/build_deps ]; then mkdir /tmp/build_deps; fi
  cd /tmp/build_deps

  # These definitions were generated with pexports on msvcr90.dll.
  # The C++ stuff at the beginning was removed. _onexit and atexit made
  # data entries.
  cat > msvcr90.def << 'THE_END'
EXPORTS
_CIacos
_CIasin
_CIatan
_CIatan2
_CIcos
_CIcosh
_CIexp
_CIfmod
_CIlog
_CIlog10
_CIpow
_CIsin
_CIsinh
_CIsqrt
_CItan
_CItanh
_CRT_RTC_INIT
_CRT_RTC_INITW
_CreateFrameInfo
_CxxThrowException
_EH_prolog
_FindAndUnlinkFrame
_Getdays
_Getmonths
_Gettnames
_HUGE DATA
_IsExceptionObjectToBeDestroyed
_NLG_Dispatch2
_NLG_Return
_NLG_Return2
_Strftime
_XcptFilter
__AdjustPointer
__BuildCatchObject
__BuildCatchObjectHelper
__CppXcptFilter
__CxxCallUnwindDelDtor
__CxxCallUnwindDtor
__CxxCallUnwindStdDelDtor
__CxxCallUnwindVecDtor
__CxxDetectRethrow
__CxxExceptionFilter
__CxxFrameHandler
__CxxFrameHandler2
__CxxFrameHandler3
__CxxLongjmpUnwind
__CxxQueryExceptionSize
__CxxRegisterExceptionObject
__CxxUnregisterExceptionObject
__DestructExceptionObject
__FrameUnwindFilter
__RTCastToVoid
__RTDynamicCast
__RTtypeid
__STRINGTOLD
__STRINGTOLD_L
__TypeMatch
___fls_getvalue@4
___fls_setvalue@8
___lc_codepage_func
___lc_collate_cp_func
___lc_handle_func
___mb_cur_max_func
___mb_cur_max_l_func
___setlc_active_func
___unguarded_readlc_active_add_func
__argc DATA
__argv DATA
__badioinfo DATA
__clean_type_info_names_internal
__control87_2
__create_locale
__crtCompareStringA
__crtCompareStringW
__crtGetLocaleInfoW
__crtGetStringTypeW
__crtLCMapStringA
__crtLCMapStringW
__daylight
__dllonexit
__doserrno
__dstbias
__fpecode
__free_locale
__get_app_type
__get_current_locale
__get_flsindex
__get_tlsindex
__getmainargs
__initenv DATA
__iob_func
__isascii
__iscsym
__iscsymf
__iswcsym
__iswcsymf
__lc_clike DATA
__lc_codepage DATA
__lc_collate_cp DATA
__lc_handle DATA
__lconv DATA
__lconv_init
__libm_sse2_acos
__libm_sse2_acosf
__libm_sse2_asin
__libm_sse2_asinf
__libm_sse2_atan
__libm_sse2_atan2
__libm_sse2_atanf
__libm_sse2_cos
__libm_sse2_cosf
__libm_sse2_exp
__libm_sse2_expf
__libm_sse2_log
__libm_sse2_log10
__libm_sse2_log10f
__libm_sse2_logf
__libm_sse2_pow
__libm_sse2_powf
__libm_sse2_sin
__libm_sse2_sinf
__libm_sse2_tan
__libm_sse2_tanf
__mb_cur_max DATA
__p___argc
__p___argv
__p___initenv
__p___mb_cur_max
__p___wargv
__p___winitenv
__p__acmdln
__p__amblksiz
__p__commode
__p__daylight
__p__dstbias
__p__environ
__p__fmode
__p__iob
__p__mbcasemap
__p__mbctype
__p__pctype
__p__pgmptr
__p__pwctype
__p__timezone
__p__tzname
__p__wcmdln
__p__wenviron
__p__wpgmptr
__pctype_func
__pioinfo DATA
__pwctype_func
__pxcptinfoptrs
__report_gsfailure
__set_app_type
__set_flsgetvalue
__setlc_active DATA
__setusermatherr
__strncnt
__swprintf_l
__sys_errlist
__sys_nerr
__threadhandle
__threadid
__timezone
__toascii
__tzname
__unDName
__unDNameEx
__unDNameHelper
__uncaught_exception
__unguarded_readlc_active DATA
__vswprintf_l
__wargv DATA
__wcserror
__wcserror_s
__wcsncnt
__wgetmainargs
__winitenv DATA
_abnormal_termination
_abs64
_access
_access_s
_acmdln DATA
_adj_fdiv_m16i
_adj_fdiv_m32
_adj_fdiv_m32i
_adj_fdiv_m64
_adj_fdiv_r
_adj_fdivr_m16i
_adj_fdivr_m32
_adj_fdivr_m32i
_adj_fdivr_m64
_adj_fpatan
_adj_fprem
_adj_fprem1
_adj_fptan
_adjust_fdiv DATA
_aexit_rtn DATA
_aligned_free
_aligned_malloc
_aligned_msize
_aligned_offset_malloc
_aligned_offset_realloc
_aligned_offset_recalloc
_aligned_realloc
_aligned_recalloc
_amsg_exit
_assert
_atodbl
_atodbl_l
_atof_l
_atoflt
_atoflt_l
_atoi64
_atoi64_l
_atoi_l
_atol_l
_atoldbl
_atoldbl_l
_beep
_beginthread
_beginthreadex
_byteswap_uint64
_byteswap_ulong
_byteswap_ushort
_c_exit
_cabs
_callnewh
_calloc_crt
_cexit
_cgets
_cgets_s
_cgetws
_cgetws_s
_chdir
_chdrive
_chgsign
_chkesp
_chmod
_chsize
_chsize_s
_clearfp
_close
_commit
_commode DATA
_configthreadlocale
_control87
_controlfp
_controlfp_s
_copysign
_cprintf
_cprintf_l
_cprintf_p
_cprintf_p_l
_cprintf_s
_cprintf_s_l
_cputs
_cputws
_creat
_create_locale
_crt_debugger_hook
_cscanf
_cscanf_l
_cscanf_s
_cscanf_s_l
_ctime32
_ctime32_s
_ctime64
_ctime64_s
_cwait
_cwprintf
_cwprintf_l
_cwprintf_p
_cwprintf_p_l
_cwprintf_s
_cwprintf_s_l
_cwscanf
_cwscanf_l
_cwscanf_s
_cwscanf_s_l
_daylight DATA
_decode_pointer
_difftime32
_difftime64
_dosmaperr
_dstbias DATA
_dup
_dup2
_dupenv_s
_ecvt
_ecvt_s
_encode_pointer
_encoded_null
_endthread
_endthreadex
_environ DATA
_eof
_errno
_except_handler2
_except_handler3
_except_handler4_common
_execl
_execle
_execlp
_execlpe
_execv
_execve
_execvp
_execvpe
_exit
_expand
_fclose_nolock
_fcloseall
_fcvt
_fcvt_s
_fdopen
_fflush_nolock
_fgetchar
_fgetwc_nolock
_fgetwchar
_filbuf
_filelength
_filelengthi64
_fileno
_findclose
_findfirst32
_findfirst32i64
_findfirst64
_findfirst64i32
_findnext32
_findnext32i64
_findnext64
_findnext64i32
_finite
_flsbuf
_flushall
_fmode DATA
_fpclass
_fpieee_flt
_fpreset
_fprintf_l
_fprintf_p
_fprintf_p_l
_fprintf_s_l
_fputchar
_fputwc_nolock
_fputwchar
_fread_nolock
_fread_nolock_s
_free_locale
_freea
_freea_s
_freefls
_fscanf_l
_fscanf_s_l
_fseek_nolock
_fseeki64
_fseeki64_nolock
_fsopen
_fstat32
_fstat32i64
_fstat64
_fstat64i32
_ftell_nolock
_ftelli64
_ftelli64_nolock
_ftime32
_ftime32_s
_ftime64
_ftime64_s
_ftol
_fullpath
_futime32
_futime64
_fwprintf_l
_fwprintf_p
_fwprintf_p_l
_fwprintf_s_l
_fwrite_nolock
_fwscanf_l
_fwscanf_s_l
_gcvt
_gcvt_s
_get_amblksiz
_get_current_locale
_get_daylight
_get_doserrno
_get_dstbias
_get_errno
_get_fmode
_get_heap_handle
_get_invalid_parameter_handler
_get_osfhandle
_get_output_format
_get_pgmptr
_get_printf_count_output
_get_purecall_handler
_get_sbh_threshold
_get_terminate
_get_timezone
_get_tzname
_get_unexpected
_get_wpgmptr
_getc_nolock
_getch
_getch_nolock
_getche
_getche_nolock
_getcwd
_getdcwd
_getdcwd_nolock
_getdiskfree
_getdllprocaddr
_getdrive
_getdrives
_getmaxstdio
_getmbcp
_getpid
_getptd
_getsystime
_getw
_getwch
_getwch_nolock
_getwche
_getwche_nolock
_getws
_getws_s
_global_unwind2
_gmtime32
_gmtime32_s
_gmtime64
_gmtime64_s
_heapadd
_heapchk
_heapmin
_heapset
_heapused
_heapwalk
_hypot
_hypotf
_i64toa
_i64toa_s
_i64tow
_i64tow_s
_initptd
_initterm
_initterm_e
_inp
_inpd
_inpw
_invalid_parameter
_invalid_parameter_noinfo
_invoke_watson
_iob DATA
_isalnum_l
_isalpha_l
_isatty
_iscntrl_l
_isctype
_isctype_l
_isdigit_l
_isgraph_l
_isleadbyte_l
_islower_l
_ismbbalnum
_ismbbalnum_l
_ismbbalpha
_ismbbalpha_l
_ismbbgraph
_ismbbgraph_l
_ismbbkalnum
_ismbbkalnum_l
_ismbbkana
_ismbbkana_l
_ismbbkprint
_ismbbkprint_l
_ismbbkpunct
_ismbbkpunct_l
_ismbblead
_ismbblead_l
_ismbbprint
_ismbbprint_l
_ismbbpunct
_ismbbpunct_l
_ismbbtrail
_ismbbtrail_l
_ismbcalnum
_ismbcalnum_l
_ismbcalpha
_ismbcalpha_l
_ismbcdigit
_ismbcdigit_l
_ismbcgraph
_ismbcgraph_l
_ismbchira
_ismbchira_l
_ismbckata
_ismbckata_l
_ismbcl0
_ismbcl0_l
_ismbcl1
_ismbcl1_l
_ismbcl2
_ismbcl2_l
_ismbclegal
_ismbclegal_l
_ismbclower
_ismbclower_l
_ismbcprint
_ismbcprint_l
_ismbcpunct
_ismbcpunct_l
_ismbcspace
_ismbcspace_l
_ismbcsymbol
_ismbcsymbol_l
_ismbcupper
_ismbcupper_l
_ismbslead
_ismbslead_l
_ismbstrail
_ismbstrail_l
_isnan
_isprint_l
_ispunct_l
_isspace_l
_isupper_l
_iswalnum_l
_iswalpha_l
_iswcntrl_l
_iswcsym_l
_iswcsymf_l
_iswctype_l
_iswdigit_l
_iswgraph_l
_iswlower_l
_iswprint_l
_iswpunct_l
_iswspace_l
_iswupper_l
_iswxdigit_l
_isxdigit_l
_itoa
_itoa_s
_itow
_itow_s
_j0
_j1
_jn
_kbhit
_lfind
_lfind_s
_loaddll
_local_unwind2
_local_unwind4
_localtime32
_localtime32_s
_localtime64
_localtime64_s
_lock
_lock_file
_locking
_logb
_longjmpex
_lrotl
_lrotr
_lsearch
_lsearch_s
_lseek
_lseeki64
_ltoa
_ltoa_s
_ltow
_ltow_s
_makepath
_makepath_s
_malloc_crt
_mbbtombc
_mbbtombc_l
_mbbtype
_mbbtype_l
_mbcasemap DATA
_mbccpy
_mbccpy_l
_mbccpy_s
_mbccpy_s_l
_mbcjistojms
_mbcjistojms_l
_mbcjmstojis
_mbcjmstojis_l
_mbclen
_mbclen_l
_mbctohira
_mbctohira_l
_mbctokata
_mbctokata_l
_mbctolower
_mbctolower_l
_mbctombb
_mbctombb_l
_mbctoupper
_mbctoupper_l
_mbctype DATA
_mblen_l
_mbsbtype
_mbsbtype_l
_mbscat_s
_mbscat_s_l
_mbschr
_mbschr_l
_mbscmp
_mbscmp_l
_mbscoll
_mbscoll_l
_mbscpy_s
_mbscpy_s_l
_mbscspn
_mbscspn_l
_mbsdec
_mbsdec_l
_mbsicmp
_mbsicmp_l
_mbsicoll
_mbsicoll_l
_mbsinc
_mbsinc_l
_mbslen
_mbslen_l
_mbslwr
_mbslwr_l
_mbslwr_s
_mbslwr_s_l
_mbsnbcat
_mbsnbcat_l
_mbsnbcat_s
_mbsnbcat_s_l
_mbsnbcmp
_mbsnbcmp_l
_mbsnbcnt
_mbsnbcnt_l
_mbsnbcoll
_mbsnbcoll_l
_mbsnbcpy
_mbsnbcpy_l
_mbsnbcpy_s
_mbsnbcpy_s_l
_mbsnbicmp
_mbsnbicmp_l
_mbsnbicoll
_mbsnbicoll_l
_mbsnbset
_mbsnbset_l
_mbsnbset_s
_mbsnbset_s_l
_mbsncat
_mbsncat_l
_mbsncat_s
_mbsncat_s_l
_mbsnccnt
_mbsnccnt_l
_mbsncmp
_mbsncmp_l
_mbsncoll
_mbsncoll_l
_mbsncpy
_mbsncpy_l
_mbsncpy_s
_mbsncpy_s_l
_mbsnextc
_mbsnextc_l
_mbsnicmp
_mbsnicmp_l
_mbsnicoll
_mbsnicoll_l
_mbsninc
_mbsninc_l
_mbsnlen
_mbsnlen_l
_mbsnset
_mbsnset_l
_mbsnset_s
_mbsnset_s_l
_mbspbrk
_mbspbrk_l
_mbsrchr
_mbsrchr_l
_mbsrev
_mbsrev_l
_mbsset
_mbsset_l
_mbsset_s
_mbsset_s_l
_mbsspn
_mbsspn_l
_mbsspnp
_mbsspnp_l
_mbsstr
_mbsstr_l
_mbstok
_mbstok_l
_mbstok_s
_mbstok_s_l
_mbstowcs_l
_mbstowcs_s_l
_mbstrlen
_mbstrlen_l
_mbstrnlen
_mbstrnlen_l
_mbsupr
_mbsupr_l
_mbsupr_s
_mbsupr_s_l
_mbtowc_l
_memccpy
_memicmp
_memicmp_l
_mkdir
_mkgmtime32
_mkgmtime64
_mktemp
_mktemp_s
_mktime32
_mktime64
_msize
_nextafter
_onexit DATA
_open
_open_osfhandle
_outp
_outpd
_outpw
_pclose
_pctype DATA
_pgmptr DATA
_pipe
_popen
_printf_l
_printf_p
_printf_p_l
_printf_s_l
_purecall
_putch
_putch_nolock
_putenv
_putenv_s
_putw
_putwch
_putwch_nolock
_putws
_pwctype DATA
_read
_realloc_crt
_recalloc
_recalloc_crt
_resetstkoflw
_rmdir
_rmtmp
_rotl
_rotl64
_rotr
_rotr64
_safe_fdiv
_safe_fdivr
_safe_fprem
_safe_fprem1
_scalb
_scanf_l
_scanf_s_l
_scprintf
_scprintf_l
_scprintf_p
_scprintf_p_l
_scwprintf
_scwprintf_l
_scwprintf_p
_scwprintf_p_l
_searchenv
_searchenv_s
_seh_longjmp_unwind
_seh_longjmp_unwind4
_set_SSE2_enable
_set_abort_behavior
_set_amblksiz
_set_controlfp
_set_doserrno
_set_errno
_set_error_mode
_set_fmode
_set_invalid_parameter_handler
_set_malloc_crt_max_wait
_set_output_format
_set_printf_count_output
_set_purecall_handler
_set_sbh_threshold
_seterrormode
_setjmp
_setjmp3
_setmaxstdio
_setmbcp
_setmode
_setsystime
_sleep
_snprintf
_snprintf_c
_snprintf_c_l
_snprintf_l
_snprintf_s
_snprintf_s_l
_snscanf
_snscanf_l
_snscanf_s
_snscanf_s_l
_snwprintf
_snwprintf_l
_snwprintf_s
_snwprintf_s_l
_snwscanf
_snwscanf_l
_snwscanf_s
_snwscanf_s_l
_sopen
_sopen_s
_spawnl
_spawnle
_spawnlp
_spawnlpe
_spawnv
_spawnve
_spawnvp
_spawnvpe
_splitpath
_splitpath_s
_sprintf_l
_sprintf_p
_sprintf_p_l
_sprintf_s_l
_sscanf_l
_sscanf_s_l
_stat32
_stat32i64
_stat64
_stat64i32
_statusfp
_statusfp2
_strcoll_l
_strdate
_strdate_s
_strdup
_strerror
_strerror_s
_strftime_l
_stricmp
_stricmp_l
_stricoll
_stricoll_l
_strlwr
_strlwr_l
_strlwr_s
_strlwr_s_l
_strncoll
_strncoll_l
_strnicmp
_strnicmp_l
_strnicoll
_strnicoll_l
_strnset
_strnset_s
_strrev
_strset
_strset_s
_strtime
_strtime_s
_strtod_l
_strtoi64
_strtoi64_l
_strtol_l
_strtoui64
_strtoui64_l
_strtoul_l
_strupr
_strupr_l
_strupr_s
_strupr_s_l
_strxfrm_l
_swab
_swprintf
_swprintf_c
_swprintf_c_l
_swprintf_p
_swprintf_p_l
_swprintf_s_l
_swscanf_l
_swscanf_s_l
_sys_errlist DATA
_sys_nerr DATA
_tell
_telli64
_tempnam
_time32
_time64
_timezone DATA
_tolower
_tolower_l
_toupper
_toupper_l
_towlower_l
_towupper_l
_tzname DATA
_tzset
_ui64toa
_ui64toa_s
_ui64tow
_ui64tow_s
_ultoa
_ultoa_s
_ultow
_ultow_s
_umask
_umask_s
_ungetc_nolock
_ungetch
_ungetch_nolock
_ungetwc_nolock
_ungetwch
_ungetwch_nolock
_unlink
_unloaddll
_unlock
_unlock_file
_utime32
_utime64
_vcprintf
_vcprintf_l
_vcprintf_p
_vcprintf_p_l
_vcprintf_s
_vcprintf_s_l
_vcwprintf
_vcwprintf_l
_vcwprintf_p
_vcwprintf_p_l
_vcwprintf_s
_vcwprintf_s_l
_vfprintf_l
_vfprintf_p
_vfprintf_p_l
_vfprintf_s_l
_vfwprintf_l
_vfwprintf_p
_vfwprintf_p_l
_vfwprintf_s_l
_vprintf_l
_vprintf_p
_vprintf_p_l
_vprintf_s_l
_vscprintf
_vscprintf_l
_vscprintf_p
_vscprintf_p_l
_vscwprintf
_vscwprintf_l
_vscwprintf_p
_vscwprintf_p_l
_vsnprintf
_vsnprintf_c
_vsnprintf_c_l
_vsnprintf_l
_vsnprintf_s
_vsnprintf_s_l
_vsnwprintf
_vsnwprintf_l
_vsnwprintf_s
_vsnwprintf_s_l
_vsprintf_l
_vsprintf_p
_vsprintf_p_l
_vsprintf_s_l
_vswprintf
_vswprintf_c
_vswprintf_c_l
_vswprintf_l
_vswprintf_p
_vswprintf_p_l
_vswprintf_s_l
_vwprintf_l
_vwprintf_p
_vwprintf_p_l
_vwprintf_s_l
_waccess
_waccess_s
_wasctime
_wasctime_s
_wassert
_wchdir
_wchmod
_wcmdln DATA
_wcreat
_wcscoll_l
_wcsdup
_wcserror
_wcserror_s
_wcsftime_l
_wcsicmp
_wcsicmp_l
_wcsicoll
_wcsicoll_l
_wcslwr
_wcslwr_l
_wcslwr_s
_wcslwr_s_l
_wcsncoll
_wcsncoll_l
_wcsnicmp
_wcsnicmp_l
_wcsnicoll
_wcsnicoll_l
_wcsnset
_wcsnset_s
_wcsrev
_wcsset
_wcsset_s
_wcstod_l
_wcstoi64
_wcstoi64_l
_wcstol_l
_wcstombs_l
_wcstombs_s_l
_wcstoui64
_wcstoui64_l
_wcstoul_l
_wcsupr
_wcsupr_l
_wcsupr_s
_wcsupr_s_l
_wcsxfrm_l
_wctime32
_wctime32_s
_wctime64
_wctime64_s
_wctomb_l
_wctomb_s_l
_wctype
_wdupenv_s
_wenviron DATA
_wexecl
_wexecle
_wexeclp
_wexeclpe
_wexecv
_wexecve
_wexecvp
_wexecvpe
_wfdopen
_wfindfirst32
_wfindfirst32i64
_wfindfirst64
_wfindfirst64i32
_wfindnext32
_wfindnext32i64
_wfindnext64
_wfindnext64i32
_wfopen
_wfopen_s
_wfreopen
_wfreopen_s
_wfsopen
_wfullpath
_wgetcwd
_wgetdcwd
_wgetdcwd_nolock
_wgetenv
_wgetenv_s
_wmakepath
_wmakepath_s
_wmkdir
_wmktemp
_wmktemp_s
_wopen
_wperror
_wpgmptr DATA
_wpopen
_wprintf_l
_wprintf_p
_wprintf_p_l
_wprintf_s_l
_wputenv
_wputenv_s
_wremove
_wrename
_write
_wrmdir
_wscanf_l
_wscanf_s_l
_wsearchenv
_wsearchenv_s
_wsetlocale
_wsopen
_wsopen_s
_wspawnl
_wspawnle
_wspawnlp
_wspawnlpe
_wspawnv
_wspawnve
_wspawnvp
_wspawnvpe
_wsplitpath
_wsplitpath_s
_wstat32
_wstat32i64
_wstat64
_wstat64i32
_wstrdate
_wstrdate_s
_wstrtime
_wstrtime_s
_wsystem
_wtempnam
_wtmpnam
_wtmpnam_s
_wtof
_wtof_l
_wtoi
_wtoi64
_wtoi64_l
_wtoi_l
_wtol
_wtol_l
_wunlink
_wutime32
_wutime64
_y0
_y1
_yn
abort
abs
acos
asctime
asctime_s
asin
atan
atan2
atexit DATA
atof
atoi
atol
bsearch
bsearch_s
btowc
calloc
ceil
clearerr
clearerr_s
clock
cos
cosh
div
exit
exp
fabs
fclose
feof
ferror
fflush
fgetc
fgetpos
fgets
fgetwc
fgetws
floor
fmod
fopen
fopen_s
fprintf
fprintf_s
fputc
fputs
fputwc
fputws
fread
fread_s
free
freopen
freopen_s
frexp
fscanf
fscanf_s
fseek
fsetpos
ftell
fwprintf
fwprintf_s
fwrite
fwscanf
fwscanf_s
getc
getchar
getenv
getenv_s
gets
gets_s
getwc
getwchar
is_wctype
isalnum
isalpha
iscntrl
isdigit
isgraph
isleadbyte
islower
isprint
ispunct
isspace
isupper
iswalnum
iswalpha
iswascii
iswcntrl
iswctype
iswdigit
iswgraph
iswlower
iswprint
iswpunct
iswspace
iswupper
iswxdigit
isxdigit
labs
ldexp
ldiv
localeconv
log
log10
longjmp
malloc
mblen
mbrlen
mbrtowc
mbsrtowcs
mbsrtowcs_s
mbstowcs
mbstowcs_s
mbtowc
memchr
memcmp
memcpy
memcpy_s
memmove
memmove_s
memset
modf
perror
pow
printf
printf_s
putc
putchar
puts
putwc
putwchar
qsort
qsort_s
raise
rand
rand_s
realloc
remove
rename
rewind
scanf
scanf_s
setbuf
setlocale
setvbuf
signal
sin
sinh
sprintf
sprintf_s
sqrt
srand
sscanf
sscanf_s
strcat
strcat_s
strchr
strcmp
strcoll
strcpy
strcpy_s
strcspn
strerror
strerror_s
strftime
strlen
strncat
strncat_s
strncmp
strncpy
strncpy_s
strnlen
strpbrk
strrchr
strspn
strstr
strtod
strtok
strtok_s
strtol
strtoul
strxfrm
swprintf_s
swscanf
swscanf_s
system
tan
tanh
tmpfile
tmpfile_s
tmpnam
tmpnam_s
tolower
toupper
towlower
towupper
ungetc
ungetwc
vfprintf
vfprintf_s
vfwprintf
vfwprintf_s
vprintf
vprintf_s
vsprintf
vsprintf_s
vswprintf_s
vwprintf
vwprintf_s
wcrtomb
wcrtomb_s
wcscat
wcscat_s
wcschr
wcscmp
wcscoll
wcscpy
wcscpy_s
wcscspn
wcsftime
wcslen
wcsncat
wcsncat_s
wcsncmp
wcsncpy
wcsncpy_s
wcsnlen
wcspbrk
wcsrchr
wcsrtombs
wcsrtombs_s
wcsspn
wcsstr
wcstod
wcstok
wcstok_s
wcstol
wcstombs
wcstombs_s
wcstoul
wcsxfrm
wctob
wctomb
wctomb_s
wprintf
wprintf_s
wscanf
wscanf_s
THE_END

  # Provide the gmtime stub required by PNG.
  cat > gmtime.c << 'THE_END'
/* Stub function for gmtime.
 * This is an inline function in Visual C 2008 so is missing from msvcr90.dll
 */
#include <time.h>

struct tm* _gmtime32(const time_t *timer);

struct tm* gmtime(const time_t *timer)
{
    return _gmtime32(timer);                                    
}
THE_END

  # Provide the _ftime stub required by numpy.random.mtrand.
  cat > _ftime.c << 'THE_END'
/* Stub function for _ftime.
 * This is an inline function in Visual C 2008 so is missing from msvcr90.dll
 */
#include <sys/types.h>
#include <sys/timeb.h>

void _ftime32(struct _timeb *timeptr);

void _ftime(struct _timeb *timeptr)
{
    _ftime32(timeptr);                                    
}
THE_END

  # Provide the time stub required by Numeric.RNG.
  cat > time.c << 'THE_END'
/* Stub function for time.
 * This is an inline function in Visual C 2008 so is missing from msvcr90.dll
 */
#include <time.h>

time_t _time32(time_t *timer);

time_t time(time_t *timer)
{
    return _time32(timer);                                    
}
THE_END

  gcc -c -O2 gmtime.c _ftime.c time.c
  dlltool -d msvcr90.def -D msvcr90.dll -l libmsvcr90.dll.a
  ar rc libmsvcr90.dll.a gmtime.o _ftime.o time.o
  ranlib libmsvcr90.dll.a
  cp -f libmsvcr90.dll.a "$DBMSVCR90"
  mv -f libmsvcr90.dll.a "$DBMSVCR90/libmsvcrt.dll.a"
  gcc -c -g gmtime.c
  dlltool -d msvcr90.def -D msvcr90d.dll -l libmsvcr90d.dll.a
  ar rc libmsvcr90d.dll.a gmtime.o
  ranlib libmsvcr90d.dll.a
  cp -f libmsvcr90d.dll.a "$DBMSVCR90"
  mv -f libmsvcr90d.dll.a "$DBMSVCR90/libmsvcrtd.dll.a"
  
  # These definitions are taken from mingw-runtime-3.12 .
  # The file was generated with the following command:
  #
  # gcc -DRUNTIME=msvcrt -D__FILENAME__=moldname-msvcrt.def
  #   -D__MSVCRT__ -C -E -P -xc-header moldname.def.in >moldname-msvcrt.def
  # It then had fstat deleted to match with msvcr90.dll.
  cat > moldname-msvcrt.def << 'THE_END'
EXPORTS
access
chdir
chmod
chsize
close
creat
cwait

daylight DATA

dup
dup2
ecvt
eof
execl
execle
execlp
execlpe
execv
execve
execvp
execvpe
fcvt
fdopen
fgetchar
fgetwchar
filelength
fileno
; Alias fpreset is set in CRT_fp10,c and CRT_fp8.c.
; fpreset
fputchar
fputwchar
ftime
gcvt
getch
getche
getcwd
getpid
getw
heapwalk
isatty
itoa
kbhit
lfind
lsearch
lseek
ltoa
memccpy
memicmp
mkdir
mktemp
open
pclose
popen
putch
putenv
putw
read
rmdir
rmtmp
searchenv
setmode
sopen
spawnl
spawnle
spawnlp
spawnlpe
spawnv
spawnve
spawnvp
spawnvpe
stat
strcmpi
strdup
stricmp
stricoll
strlwr
strnicmp
strnset
strrev
strset
strupr
swab
tell
tempnam

timezone DATA

; export tzname for both. See <time.h>
tzname DATA
tzset
umask
ungetch
unlink
utime
wcsdup
wcsicmp
wcsicoll
wcslwr
wcsnicmp
wcsnset
wcsrev
wcsset
wcsupr

wpopen

write
; non-ANSI functions declared in math.h
j0
j1
jn
y0
y1
yn
chgsign
scalb
finite
fpclass
; C99 functions
cabs
hypot
logb
nextafter
THE_END

  # Provide the fstat stub required by TIFF.
  cat > fstat.c << 'THE_END'
/* Stub function for fstat.
 * This is an inlined functions in Visual C 2008 so is missing from msvcr90.dll
 */
#include <sys/stat.h>

int _fstat32(int fd, struct stat *buffer);

int fstat(int fd, struct stat *buffer)
{
    return _fstat32(fd, buffer);
}
THE_END

  mkdir -p "$DBMSVCR90"
  gcc -c -O2 fstat.c
  ar x /mingw/lib/libmoldname90.a $OBJS
  dlltool --as as -k -U \
     --dllname msvcr90.dll \
     --def moldname-msvcrt.def \
     --output-lib libmoldname.dll.a
  ar rc libmoldname.dll.a $OBJS fstat.o
  ranlib libmoldname.dll.a
  mv -f libmoldname.dll.a "$DBMSVCR90"
  gcc -c -g fstat.c
  ar x /mingw/lib/libmoldname90d.a $OBJS
  dlltool --as as -k -U \
     --dllname msvcr90.dll \
     --def moldname-msvcrt.def \
     --output-lib libmoldnamed.dll.a
  ar rc libmoldnamed.dll.a $OBJS fstat.o
  ranlib libmoldnamed.dll.a
  mv -f libmoldnamed.dll.a "$DBMSVCR90"
  rm -f ./*
  cd "$OLDPWD"
  rmdir /tmp/build_deps
fi
""")

if __name__ == '__main__':
    sys.exit(main(dependencies, msvcr90_prep, msys_prep))
