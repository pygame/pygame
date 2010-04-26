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

import sys, os, re
import zipfile
import tarfile
# Python 3.x workarounds for the changed urllib and stringio modules.
if sys.version_info[0] >= 3:
    import urllib.parse as urlparse
    import urllib.request as urllib2
    import io
else:
    import urlparse
    import urllib2
    try:
        import cStringIO as io
    except ImportError:
        import StringIO as io

def _get_stringio (data):
    """_get_stringio (data) -> StringIO or BytesIO

    Returns a StringIO instance for Python < 3 or a BytesIO or Python > 3.
    """
    iodata = None
    if sys.version_info[0] >= 3:
        iodata = io.BytesIO (data)
    else:
        iodata = io.StringIO (data)
    return iodata

def open_zipfile (archive, filename, dir=None):
    """open_zipfile (archive, filename, dir=None) -> StringIO or BytesIO
    
    Opens and reads a certain file from a ZIP archive.
    
    Opens and reads a certain file from a ZIP archive. The result is returned
    as StringIO stream. *filename* can be a relative or absolute path within
    the ZIP archive. The optional *dir* argument can be used to supply a
    relative directory path, under which *filename* will be tried to retrieved.
    
    If the *filename* could not be found or an error occured on reading it,
    None will be returned.
    
    Raises a TypeError, if *archive* is not a valid ZIP archive.
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
        data = _get_stringio (dmpdata.read ())
    finally:
        if opened:
            archive.close ()
    return data

def open_tarfile (archive, filename, dir=None, type=None):
    """open_tarfile (archive, filename, dir=None, type=None) -> StringIO or BytesIO
    
    Opens and reads a certain file from a TAR archive.
    
    Opens and reads a certain file from a TAR archive. The result is returned
    as StringIO stream. *filename* can be a relative or absolute path within
    the TAR archive. The optional *dir* argument can be used to supply a
    relative directory path, under which *filename* will be tried to retrieved.

    *type* is used to supply additional compression information, in case the
    system cannot determine the compression type itself, and can be either
    'gz' for gzip compression or 'bz2' for bzip2 compression.
    
    Note:
      
      If *type* is supplied, the compreesion mode will be enforced for opening
      and reading.
    
    If the *filename* could not be found or an error occured on reading it,
    None will be returned.
    
    Raises a TypeError, if *archive* is not a valid TAR archive or if *type*
    is not a valid value of ('gz', 'bz2').
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
        data = _get_stringio (dmpdata.read ())
    finally:
        if opened:
            archive.close ()
    return data
    
def open_url (filename, basepath=None):
    """open_url (filename, basepath=None) -> file
    
    Opens and reads a certain file from a web or remote location.
    
    Opens and reads a certain file from a web or remote location. This function
    utilizes the urllib2 module, which means that it is restricted to the types
    of remote locations supported by urllib2.
    
    *basepath* can be used to supply an additional location prefix.
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
        """_scanzip (filename) -> None
        
        Scans the passed ZIP archive and indexes all the files contained by it.
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
        """_scantar (filename, type=None) -> None
        
        Scans the passed TAR archive and indexes all the files contained by it.
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
        """get (filename) -> StringIO or BytesIO
       
        Gets a specific file from the Resources.
        
        Raises a KeyError, if *filename* could not be found.
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
        data = _get_stringio (dmpdata.read ())
        dmpdata.close ()
        return data
    
    def get_filelike (self, filename):
        """get_filelike (filename) -> file or StringIO or BytesIO
        
        Like get(), but tries to return the original file handle, if possible.
        
        If the passed *filename* is only available within an archive, a
        StringIO instance will be returned.
        
        Raises a KeyError, if *filename* could not be found.
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
        
        Gets the path of the passed filename.
        
        If *filename* is only available within an archive, a string in the form
        'filename@archivename' will be returned.
        
        Raises a KeyError, if *filename* could not be found.
        """
        archive, type, pathname = self.files[filename]
        if archive:
            return '%s@%s' % (pathname, archive)
        return pathname

    def scan (self, path, excludepattern=None):
        """scan (path) -> None
        
        Scans a path and adds all found files to the Resource container.
        
        Scans a path and adds all found files to the Resource container. If a
        file is a supported (ZIP or TAR) archive, its contents will be indexed
        and added automatically.
        
        *excludepattern* can be a regular expression to skip files, which match
        the pattern.
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
