import msys
from distutils.extension import read_setup_file

import os
import re
import sys

prebuilt_dir = 'prebuilt'
lib_subdir = 'lib'

class MakePrebuiltError(Exception):
    pass

def file_copy(src, dest):
    if os.path.isdir(dest):
        dest = os.path.join(dest, os.path.split(src)[1])
    s = open(src, 'rb')
    try:
        d = open(dest, 'wb')
        try:
            d.write(s.read())
            print "%s => %s" % (src, dest)
        finally:
            d.close()
    finally:
        s.close()

def find_import_libraries(path, roots):
    match = re.compile(r'lib(%s)\.dll\.a$' % '|'.join(roots)).match
    return [name for name in os.listdir(path) if match(name) is not None]

def copy_dir(src, dest):
    if dest == '.':
        ignore, dest = os.path.split(src)
    if src == dest:
        raise IOError("%s: Source and destination are identical" % src)
    mkdir(dest)
    for name in os.listdir(src):
        src_path = os.path.join(src, name)
        if os.path.isfile(src_path):
            file_copy(src_path, os.path.join(dest, name))

def confirm(message):
    "ask a yes/no question, return result"
    reply = msys.msys_raw_input("\n%s [Y/n]:" % message)
    if reply and reply[0].lower() == 'n':
        return 0
    return 1

created_dirs = set()

def mkdir(path):
    path = os.path.abspath(path)
    if path in created_dirs:
        pass
    elif not os.path.exists(path):
        os.mkdir(path)
        created_dirs.add(path)
    elif not os.path.isdir(path):
        raise MakePrebuiltError("%s is not a directory" % path)

def main(dest_dir=None):
    # Top level directories.
    if dest_dir is None:
        dest_dir = prebuilt_dir
    if re.match(r'([A-Za-z]:){0,1}[^"<>:|?*]+$', dest_dir) is None:
        print "Invalid directory path name %s" % dest_dir
        return 1
    dest_dir = os.path.abspath(dest_dir)
    if os.path.isdir(dest_dir):
        if not confirm("Directory %s already exists;\ncontinue" % dest_dir):
            return 1
    mkdir(dest_dir)
    m = msys.Msys()
    src_dir = os.path.join(m.msys_root, 'local')
    prebuilt_template = os.path.abspath('prebuilt-template')
    dest_lib_dir = os.path.join(dest_dir, lib_subdir)
    
    # Read setup file.
    src_file = os.path.join(prebuilt_template, 'Setup_Win.in')
    file_copy(src_file, dest_dir)
    deps =  read_setup_file(src_file)
    setup_in = open(src_file)
    match = re.compile('[A-Z_][A-Z0-9_]* *=(.*)').match
    header_dir_pat = re.compile(' -I([^ ]+)')
    lib_pat = re.compile(' -l([^ ]+)')
    macros = []
    for line in setup_in:
        matches = match(line)
        if matches is not None:
            flags = matches.group(1)
            header_dirs = header_dir_pat.findall(flags)
            libs = lib_pat.findall(flags)
            macros.append((header_dirs, libs))

    # Copy DLLs.
    src_bin_dir = os.path.join(src_dir, 'bin')
    have_dlls = set()
    for dep in deps:
        path_elements = dep.library_dirs[0].split('/')  # / required by setup.
        dll_name = path_elements[-1]
        src_dll_path = os.path.join(src_bin_dir, dll_name)
        if os.path.exists(src_dll_path):
            if path_elements[0] == '.':
                path_elements = path_elements[2:]
            else:
                path_elements = path_elements[1:]
            dest_dll_dir = dest_dir
            for dir_name in path_elements[:-1]:
                dest_dll_dir = os.path.join(dest_dll_dir, dir_name)
                mkdir(dest_dll_dir)
            file_copy(os.path.join(src_bin_dir, dll_name),
                      os.path.join(dest_dll_dir, dll_name))
            have_dlls.add(dep.name[8:])
    
    # Copy required import libraries only.
    copied_files = set()
    src_lib_dir = os.path.join(src_dir, 'lib')
    mkdir(dest_lib_dir)
    for ignore, libs in macros:
        use = False
        for lib in libs:
            if lib in have_dlls:
                use = True
                break
        if use and lib not in copied_files:
            copied_files.add(lib)
            lib_name = 'lib%s.dll.a' % lib
            src_lib_path = os.path.join(src_lib_dir, lib_name)
            if not os.path.exists(src_lib_path):
                print "Missing import library %s" % lib_name
                return 1
            file_copy(src_lib_path, os.path.join(dest_lib_dir, lib_name))

    # Copy required header directories only.
    copied_dirs = set()
    for header_dirs, libs in macros:
        use = False
        for lib in libs:
            if lib in have_dlls:
                use = True
                break
        if use:
            for header_dir in header_dirs:
                path_elements = header_dir.split('/')
                if path_elements[0] == '.':
                    path_elements = path_elements[2:]
                else:
                    path_elements = path_elements[1:]
                src_header_dir = os.path.join(src_dir, *path_elements)
                if not os.path.exists(src_header_dir):
                    print "Missing include directory %s" % src_header_dir
                    return 1
                dest_header_dir = dest_dir
                for dir_name in path_elements:
                    dest_header_dir = os.path.join(dest_header_dir, dir_name)
                    mkdir(dest_header_dir)
                if not src_header_dir in copied_dirs:
                    copy_dir(src_header_dir, dest_header_dir)
                    copied_dirs.add(src_header_dir)
    if 'SDL' in have_dlls:
        # For MSVC use SDL_config_win32.h in place of configure
        # generated SDL_config.h.
        file_copy(
            os.path.join(src_dir, 'include', 'SDL', 'SDL_config_win32.h'),
            os.path.join(dest_dir, 'include', 'SDL', 'SDL_config.h'))

    # msvcr71.dll linking support.
    src_msvcr71_dir = os.path.join(src_dir, 'lib', 'msvcr71')
    dest_msvcr71_dir = os.path.join(dest_dir, 'lib', 'msvcr71')
    copy_dir(src_msvcr71_dir, dest_msvcr71_dir)

    # Def file bat.
    make_defs = open(os.path.join(dest_lib_dir, 'MakeDefs.bat'), 'w')
    try:
        make_defs.write('@echo off\n'
                        'rem Make .def files needed for .lib file creation.\n'
                        'rem Requires pexports.exe on the search path\n'
                        'rem (found in altbinutils-pe as SourceForge,\n'
                        'rem http://sourceforge.net/projects/mingwrep/).\n\n')
        for dep in deps:
            dll_name = os.path.split(dep.library_dirs[0])[1]
            lib = dep.name[8:]
            lib_name = 'lib%s.dll.a' % lib
            if os.path.exists(os.path.join(dest_lib_dir, lib_name)):
                start = ''
            else:
                start = 'rem '
            make_defs.write('%spexports %s >%s.def\n' %
                            (start, dll_name, lib))
    finally:
        make_defs.close()

    # Lib import files bat.
    make_libs = open(os.path.join(dest_lib_dir, 'MakeLibs.bat'), 'w')
    try:
        make_libs.write('@echo off\n'
                        'rem Make .lib import libraries.\n'
                        'rem Requires Visual C++ Studio or Toolkit.\n'
                        'rem VCVARS32.BAT (VCVARS64.BAT (?) for 64 bit build)\n'
                        'rem must be run first to use LINK.EXE.\n\n')
        for dep in deps:
            dll_name = os.path.split(dep.library_dirs[0])[1]
            lib = dep.name[8:]
            lib_name = 'lib%s.dll.a' % lib
            if os.path.exists(os.path.join(dest_lib_dir, lib_name)):
                start = ''
            else:
                start = 'rem '
            make_libs.write('%sLINK.EXE /LIB /NOLOGO /DEF:%s.def /MACHINE:IX86 /OUT:%s.lib\n' %
                            (start, lib, lib))
    finally:
        make_libs.close()

    # Top level make batch file for 32 bit build.
    file_copy(os.path.join(prebuilt_template, 'Make32.bat'), dest_lib_dir)

    return 0

if __name__ =='__main__':
    dest_dir = None
    if len(sys.argv) > 1:
        dest_dir = sys.argv[1]
    try:
        sys.exit(main(dest_dir))
    except MakePrebuiltError, e:
        print "*** %s; execution halted" % e
        sys.exit(1)
