import msys
from distutils.extension import read_setup_file

import os
import re
import sys

prebuilt_dir = 'prebuilt'

def file_copy(src, dest):
    if dest == '.':
        dest = os.path.split(src)[1]
    if dest == src:
        raise IOError("%s: Source and destination are the same" % src)
    s = open(src, 'rb')
    try:
        d = open(dest, 'wb')
        try:
            d.write(s.read())
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

def mkdir(path):
    if path == '.':
        raise IOError("What the")
    if not os.path.exists(path):
        os.mkdir(path)
    elif os.path.isdir(path):
        if not confirm("Directory %s already exists; continue" % path):
            sys.exit(0)
    else:
        print "*** %s is not a directory; execution halted"
        sys.exit(1)

def main(prebuilt_dir=None):
    if prebuilt_dir is None:
        prebuilt_dir = prebuilt
    mkdir(prebuilt_dir)
    os.chdir(prebuilt_dir)
    file_copy(os.path.join('..', 'prebuilt-template', 'Setup_Win.in'), '.')
    deps = [dep for dep in read_setup_file('Setup_Win.in')
                if dep.name.startswith('COPYLIB_')]
    mkdir('lib')
    os.chdir('lib')
#    file_copy(os.path.join('..', '..', 'prebuilt-template', 'makelibs.bat'), '.')
    local_dir = os.path.join(msys.Msys().msys_root, 'local')
    src_dir_path = os.path.join(local_dir, 'bin')
    for d in deps:
        ignore, dll_file = os.path.split(d.library_dirs[0])
        try:
            file_copy(os.path.join(src_dir_path, dll_file), '.')
        except Exception:
            pass
    src_dir_path = os.path.join(local_dir, 'lib')
    import_libs = find_import_libraries(src_dir_path, [d.name[8:] for d in deps])
    for lib in import_libs:
        file_copy(os.path.join(src_dir_path, lib), '.')
    os.chdir('..')
    src_dir_path = os.path.join(local_dir, 'include')
    copy_dir(src_dir_path, '.')
    os.chdir('include')
    for d in ['SDL', 'libpng12', 'ogg', 'smpeg', 'vorbis']:
        copy_dir(os.path.join(src_dir_path, d), '.')

if __name__ =='__main__':
    prebuilt_dir = None
    if len(sys.argv) > 1:
        prebuilt_dir = sys.argv[1]
    main(prebuilt_dir)
