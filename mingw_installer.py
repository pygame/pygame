# MinGW installer.
#
# Requires:
#     httplib2    (http://code.google.com/p/httplib2/)
#     7-Zip 9.15  (http://sourceforge.net/projects/sevenzip/files/)
#
from __future__ import with_statement

import httplib2

import sys
import os
import _winreg
import re
import subprocess

# Usual MinGW directory, without drive letter.
default_mingw_subdir = 'MinGW'

packages = [
    # gcc core
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/GNU-Binutils/binutils-2.20.51/binutils-2.20.51-1-mingw32-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/RuntimeLibrary/MinGW-RT/mingwrt-3.18/mingwrt-3.18-mingw32-dll.tar.gz/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/RuntimeLibrary/MinGW-RT/mingwrt-3.18/mingwrt-3.18-mingw32-dev.tar.gz/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/RuntimeLibrary/Win32-API/w32api-3.14/w32api-3.14-mingw32-dev.tar.gz/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/mpc/mpc-0.8.1-1/mpc-0.8.1-1-mingw32-dev.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/mpc/mpc-0.8.1-1/libmpc-0.8.1-1-mingw32-dll-2.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/mpfr/mpfr-2.4.1-1/mpfr-2.4.1-1-mingw32-dev.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/mpfr/mpfr-2.4.1-1/libmpfr-2.4.1-1-mingw32-dll-1.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/gmp/gmp-5.0.1-1/gmp-5.0.1-1-mingw32-dev.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/gmp/gmp-5.0.1-1/libgmp-5.0.1-1-mingw32-dll-10.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/pthreads-w32/pthreads-w32-2.8.0-3/pthreads-w32-2.8.0-3-mingw32-dev.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/pthreads-w32/pthreads-w32-2.8.0-3/libpthread-2.8.0-3-mingw32-dll-2.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/GCC/Version4/gcc-4.5.0-1/libgomp-4.5.0-1-mingw32-dll-1.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/GCC/Version4/gcc-4.5.0-1/libssp-4.5.0-1-mingw32-dll-0.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/GCC/Version4/gcc-4.5.0-1/gcc-core-4.5.0-1-mingw32-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/GCC/Version4/gcc-4.5.0-1/libgcc-4.5.0-1-mingw32-dll-1.tar.lzma/download',

    # g++
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/GCC/Version4/gcc-4.5.0-1/gcc-c%2B%2B-4.5.0-1-mingw32-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/BaseSystem/GCC/Version4/gcc-4.5.0-1/libstdc%2B%2B-4.5.0-1-mingw32-dll-6.tar.lzma/download',

    # additional tools
    'http://sourceforge.net/projects/mingw/files/MinGW/pexports/pexports-0.44-1/pexports-0.44-1-mingw32-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/autoconf/wrapper/autoconf-7-1/autoconf-7-1-mingw32-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/autoconf/autoconf2.5/autoconf2.5-2.64-1/autoconf2.5-2.64-1-mingw32-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/libtool/libtool-2.2.7a-1/libtool-2.2.7a-1-mingw32-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/libtool/libtool-2.2.7a-1/libltdl-2.2.7a-1-mingw32-dll-7.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/automake/wrapper/automake-4-1/automake-4-1-mingw32-bin.tar.lzma/download',
    'http://sourceforge.net/projects/mingw/files/MinGW/automake/automake1.11/automake1.11-1.11-1/automake1.11-1.11-1-mingw32-bin.tar.lzma/download',
]

dx7_headers = 'http://www.mplayerhq.hu/MPlayer/contrib/win32/dx7headers.tgz'

yasm = 'http://www.tortall.net/projects/yasm/releases/yasm-1.0.1-win32.exe'

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

    def retrieve_tgz(self, url, destination=None):
        if destination is None:
            destination = ''
        file_name = encode_file_name(re.search(r'/(?P<name>[^/]+\.tgz)$',
                                               url).group('name'))
        if file_name is None:
            raise ValueError("Internel error: Unable to extract"
                             " package file name from URL %s" %
                             (url,))
        tar_file_name = file_name[:-3] + 'tar'
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
        retcode = subprocess.call(['7z', 'x', tar_file_name, '-aoa',
                                   '-o' + destination])
        if retcode:
            raise IOError("7-Zip failed to unpack tar file %s", (tar_file_name,))

    def retrieve_file(self, url, target=None):
        if target is None:
            target = encode_file_name(re.search(r'/(?P<name>[^/]+)$',
                                                url).group('name'))
        if target is None:
            raise ValueError("Internel error: Unable to extract"
                             " file name from URL %s" %
                             (url,))
        response, content = self.client.request(url, 'GET')
        if response.status != 200:
            raise IOError("HTTP code %i for URL %s" % (responce.status, url))
        f = open(target, 'wb')
        try:
            f.write(content)
        finally:
            f.close()

def install(mingw_dir):
    installer = PackageInstaller()
    try:
        os.mkdir(mingw_dir)
    except WindowsError:
        pass
    os.chdir(mingw_dir)
    with PackageLog() as log:
        for p in packages:
            if p not in log:
                print "-", p
                installer.retrieve(p)
                log.add(p)
        
        if dx7_headers not in log:
            installer.retrieve_tgz(dx7_headers, 'include')
            log.add(dx7_headers)
            
        if yasm not in log:
            installer.retrieve_file(yasm, 'bin\\yasm.exe')
            log.add(yasm)

if __name__ == '__main__':
    if len(sys.argv )== 2:
        mingw_dir = sys.argv[1]
    else:
        drive = os.path.splitdrive(os.getcwd())[0]
        mingw_dir = drive + os.path.sep + default_mingw_subdir
    install(mingw_dir)
 