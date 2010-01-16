##    pygame - Python Game Library
##    Copyright (C) 2010 Marcus von Appen
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##

##
## This file is placed under the Public Domain.
##

"""
Resource management methods.
"""

import os, re
import zipfile
import tarfile
import urlparse
import urllib2

try:
    import cStringIO as stringio
except ImportError:
    import StringIO as stringio

def open_zipfile (archive, filename, dir=None):
    """open_zipfile (archive, filename, dir=None) -> StringIO
    """
    data = None
    opened = False
    
    if not isinstance (archive, zipfile.ZipFile):
        if not zipfile.is_zipfile (archive):
            raise TypeError ("passed file does not seem to be a ZIP archive")
        else:
            archive = zipfile.ZipFile (archive, 'r')
            opened = True
    
    apath = filename
    if dir:
        apath = "%s/%s" % (dir, filename)
    
    try:
        dmpdata = archive.open (apath)
        data = stringio.StringIO (dmpdata.read ())
    finally:
        if opened:
            archive.close ()
    return data

def open_tarfile (archive, filename, dir=None, type=None):
    """open_tarfile (archive, filename, dir=None, type=None) -> StringIO
    """
    data = None
    opened = False
    
    mode = 'r'
    if type:
        if type not in ('gz', 'bz2'):
            raise TypeError ("invalid TAR compression type")
        mode = "r:%s" % type
    
    if not isinstance (archive, tarfile.TarFile):
        if not tarfile.is_tarfile (archive):
            raise TypeError ("passed file does not seem to be a TAR archive")
        else:
            archive = tarfile.open (archive, mode)
            opened = True
    
    apath = filename
    if dir:
        apath = "%s/%s" % (dir, filename)
    
    try:
        dmpdata = archive.extractfile (apath)
        data = stringio.StringIO (dmpdata.read ())
    finally:
        if opened:
            archive.close ()
    return data
    
def open_url (filename, basepath=None):
    """open_url (filename) -> file
    """
    url = filename
    if basepath:
        url = urlparse.urljoin (basepath, filename)
    return urllib2.urlopen (url)

class Resources (object):
    """Resources () -> Resources
    
    Creates a new resource container instance.
    
    The Resources class manages a set of file resources and eases accessing
    them by using relative paths, scanning archives automatically and so on.
    """
    def __init__ (self, path=None, excludepattern=None):
        self.files = {}
        if path:
            self.scan (path, excludepattern)

    def _scanzip (self, filename):
        """
        """
        if not zipfile.is_zipfile (filename):
            raise TypeError ("file '%s' is not a valid ZIP archive" % filename)
        archname = os.path.abspath (filename)
        zip = zipfile.ZipFile (filename, 'r')
        for path in zip.namelist ():
            dirname, fname = os.path.split (path)
            if fname:
                self.files[fname] = (archname, 'zip', path)
        zip.close ()
    
    def _scantar (self, filename, type=None):
        """
        """
        if not tarfile.is_tarfile (filename):
            raise TypeError ("file '%s' is not a valid TAR archive" % filename)
        mode = 'r'
        if type:
            if type not in ('gz', 'bz2'):
                raise TypeError ("invalid TAR compression type")
            mode = "r:%s" % type
        archname = os.path.abspath (filename)
        archtype = 'tar'
        if type:
            archtype = 'tar%s' % type
        tar = tarfile.open (filename, mode)
        for path in tar.getnames ():
            dirname, fname = os.path.split (path)
            self.files[fname] = (archname, archtype, path)
        tar.close ()
    
    def add (self, filename):
        """add (filename) -> None
        
        Adds a file to the Resources container.
        
        Depending on the file type (determined by the file suffix or name),
        the file will be automatically scanned (if it is an archive) or
        checked for availability (if it is a stream/network resource).
        """
        if zipfile.is_zipfile (filename):
            self.add_archive (filename)
        elif tarfile.is_tarfile (filename):
            self.add_archive (filename, 'tar')
        else:
            self.add_file (filename)

    def add_file (self, filename):
        """add_file (self, filename) -> None
        
        Adds a file to the Resources container.
        
        This will only add the passed file and do not scan an archive or check
        a stream for availability.
        """
        abspath = os.path.abspath (filename)
        dirname, fname = os.path.split (abspath)
        if not fname:
            raise ValueError ("invalid file path")
        self.files[fname] = (None, None, abspath)
    
    def add_archive (self, filename, typehint='zip'):
        """add_archive (self, filename, typehint='zip') -> None
        
        Adds an archive file to the Resources container.
        
        This will scan the passed archive and add its contents to the list
        of available resources.
        """
        if typehint == 'zip':
            self._scanzip (filename)
        elif typehint == 'tar':
            self._scantar (filename)
        elif typehint == 'tarbz2':
            self._scantar (filename, 'bz2')
        elif typehint == 'targz':
            self._scantar (filename, 'gz')
        else:
            raise ValueError ("unsupported archive type")

    def get (self, filename):
        """get (filename) -> file
       
        Gets the specified file from the Resources.
        """
        archive, type, pathname = self.files[filename]
        if archive:
            if type == 'zip':
                return open_zipfile (archive, pathname)
            elif type == 'tar':
                return open_tarfile (archive, pathname)
            elif type == 'tarbz2':
                return open_tarfile (archive, pathname, 'bz2')
            elif type == 'targz':
                return open_tarfile (archive, pathname, 'gz')
            else:
                raise ValueError ("unsupported archive type")
        dmpdata = open (pathname, 'rb')
        data = stringio.StringIO (dmpdata.read ())
        dmpdata.close ()
        return data
    
    def get_filelike (self, filename):
        """get_filelike (filename) -> file
        
        Like get(), but tries to return the original file handle, if possible.
        """
        archive, type, pathname = self.files[filename]
        if archive:
            if type == 'zip':
                return open_zipfile (archive, pathname)
            elif type == 'tar':
                return open_tarfile (archive, pathname)
            elif type == 'tarbz2':
                return open_tarfile (archive, pathname, 'bz2')
            elif type == 'targz':
                return open_tarfile (archive, pathname, 'gz')
            else:
                raise ValueError ("unsupported archive type")
        return open (pathname, 'rb')
    
    def get_path (self, filename):
        """get_path (filename) -> str
        """
        archive, type, pathname = self.files[filename]
        if archive:
            return '%s@%s' % (pathname, archive)
        return pathname

    def scan (self, path, excludepattern=None):
        """scan (path) -> None
        """
        match = None
        if excludepattern:
            match = re.compile (excludepattern).match
        join = os.path.join
        add = self.add
        abspath = os.path.abspath (path)
        for (p, dirnames, filenames) in os.walk (abspath):
            if match and match(p) is not None:
                continue
            for fname in filenames:
                add (join (p, fname))
