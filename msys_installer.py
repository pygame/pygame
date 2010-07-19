# MSYS installer.
#
# Requires:
#     httplib2    (http://code.google.com/p/httplib2/)
#     7-Zip 9.15  (http://sourceforge.net/projects/sevenzip/files/)
#
# Don't forget to add etc/fstab with the MinGW directory path added.
#
from __future__ import with_statement

import httplib2

import sys
import os
import _winreg
import re
import subprocess

# Usual MSYS directory, without drive letter.
default_msys_subdir = 'msys\\1.0'

packages = [
    'http://sourceforge.net/projects/mingw/files/MSYS/BaseSystem/msys-1.0.13-2/msysCORE-1.0.13-2-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/libiconv/libiconv-1.13.1-2/libiconv-1.13.1-2-msys-1.0.13-dll-2.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/gettext/gettext-0.17-2/libintl-0.17-2-msys-dll-8.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/bash/bash-3.1.17-3/bash-3.1.17-3-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/termcap/termcap-0.20050421_1-2/termcap-0.20050421_1-2-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/termcap/termcap-0.20050421_1-2/libtermcap-0.20050421_1-2-msys-1.0.13-dll-0.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/regex/regex-1.20090805-2/libregex-1.20090805-2-msys-1.0.13-dll-1.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/make/make-3.81-3/make-3.81-3-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/sed/sed-4.2.1-2/sed-4.2.1-2-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/grep/grep-2.5.4-2/grep-2.5.4-2-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/gawk/gawk-3.1.7-2/gawk-3.1.7-2-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/m4/m4-1.4.13-1/m4-1.4.13-1-msys-1.0.11-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/cvs/cvs-1.12.13-2/cvs-1.12.13-2-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/crypt/crypt-1.1_1-3/libcrypt-1.1_1-3-msys-1.0.13-dll-0.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/perl/perl-5.6.1_2-2/perl-5.6.1_2-2-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/zlib/zlib-1.2.3-2/zlib-1.2.3-2-msys-1.0.13-dll.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/gdbm/gdbm-1.8.3-3/libgdbm-1.8.3-3-msys-1.0.13-dll-3.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/coreutils/coreutils-5.97-3/coreutils-5.97-3-msys-1.0.13-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/coreutils/coreutils-5.97-3/coreutils-5.97-3-msys-1.0.13-ext.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MSYS/diffutils/diffutils-2.8.7.20071206cvs-3/diffutils-2.8.7.20071206cvs-3-msys-1.0.13-bin.tar.lzma/download',
]

def find_7zip_registry():
    """Return the path of the 7-Zip binaries
    
    Look up the location of the 7-Zip directory in the registry.
    Raise LookupError if not found.
    
    """
    
    subkey = 'SOFTWARE\\7-Zip' 
    
    try:
        key = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, subkey)
        try:
            return _winreg.QueryValueEx(key, 'path')[0].encode()
        finally:
            key.Close()
    except WindowsError:
        raise LookupError("7-Zip not found in registry")

class PackageLog(object):
    log_name = 'retrieved.txt'
    def __init__(self):
        try:
            log = open(self.log_name, 'r')
        except IOError:
            self.retrieved = []
            self.lookup = set()
        else:
            self.retrieved = [p[0:-1] for p in log if p.strip()]
            self.lookup = set(self.retrieved)
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with open(self.log_name, 'w') as log:
            for ln in self.retrieved:
                log.write(ln)
                log.write('\n')

    def __contains__(self, package):
        return package in self.lookup
        
    def add(self, package):
        self.retrieved.append(package)
        self.lookup.add(package)

encode_file_name_pattern = re.compile(r'%[a-fA-F0-9]{2}')

def encode_file_name(s):
    def encode(match):
        return chr(int(match.group(0)[1:], 16))
        
    return encode_file_name_pattern.sub(encode, s)

class PackageInstaller(object):
    filename_pattern = re.compile(r'/(?P<name>[^/]+)/download$')
    tar_filename_pattern = re.compile(r'(?P<name>.+\.tar)\.[a-z]+$')
    have_7zip = False

    def __init__(self):
        self.client = httplib2.Http()
        if not self.have_7zip:
            os.environ['path'] = os.environ['path'] + ';' + find_7zip_registry()
            self.have_7zip = True
        
    def retrieve(self, url):

        file_name = encode_file_name(self.filename_pattern.search(url).group('name'))
        if file_name is None:
            raise ValueError("Internel error: Unable to extract"
                             " package file name from URL %s" %
                             (url,))
        tar_file_name = self.tar_filename_pattern.match(file_name).group('name')
        if tar_file_name is None:
            raise ValueError("Internel error: Unable to extract tar file name"
                             " from package file name %s" % (tar_file_name,))
        response, content = self.client.request(url, 'GET')
        if response.status != 200:
            raise IOError("HTTP code %i for URL %s" % (responce.status, url))
        f = open(file_name, 'wb')
        try:
            f.write(content)
        finally:
            f.close()
        retcode = subprocess.call(['7z', 'x', file_name, '-aoa'])
        if retcode:
            raise IOError("7-Zip failed to decompress %s" % (file_name,))
        retcode = subprocess.call(['7z', 'x', tar_file_name, '-aoa'])
        if retcode:
            raise IOError("7-Zip failed to unpack tar file %s", (tar_file_name,))

def install(msys_dir):
    installer = PackageInstaller()
    try:
        os.makedirs(msys_dir)
    except WindowsError:
        pass
    os.chdir(msys_dir)
    with PackageLog() as log:
        for p in packages:
            if p not in log:
                print "-", p
                installer.retrieve(p)
                log.add(p)
    
if __name__ == '__main__':
    if len(sys.argv )== 2:
        msys_dir = sys.argv[1]
    else:
        drive = os.path.splitdrive(os.getcwd())[0]
        msys_dir = drive + os.path.sep + default_msys_subdir
    install(msys_dir)
 