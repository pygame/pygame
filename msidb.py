# -*- coding: UTF-8 -*-

# Module msidb.py
#
# Copyright: Lenard Lindstrom, 2008
# <len-l@telus.net>
#
# This module is licensed under the GNU LESSER GENERAL PUBLIC LICENSE,
# Version 2.1, February 1999. Keep with the associated license LGPL.

"""A partial DB-API 2.0 interface for msi databases"""

from ctypes import (c_char, c_int, c_uint, c_ulong,
                    c_char_p, c_wchar_p, c_void_p, POINTER,
                    byref, windll, create_string_buffer, cast)
import itertools
import re
import weakref

TCHAR_TYPE = 'A'
if TCHAR_TYPE == 'A':
    TStr = str
    Tchar_p = c_char_p
elif TCHAR_TYPE == 'W':
    TStr = unicode
    Tchar_p = c_wchar_p
else:
    raise AssertionError("Unsupported character type '%s'" % TCHAR_TYPE)
NULL_TSTR = cast(0, Tchar_p)
P_ulong = POINTER(c_ulong)
Handle = c_void_p
PHandle = POINTER(Handle)
MSICOLINFO = c_int
MSICOLINFO_NAMES = 0
MSICOLINFO_TYPES = 1
MSI_NULL_INTEGER = -0x80000000  # integer value reserved for null

ERROR_SUCCESS = 0  # The operation completed successfully.
ERROR_INVALID_HANDLE = 6  # The handle is invalid.
ERROR_INVALID_PARAMETER = 87  # The parameter is incorrect.
ERROR_MORE_DATA = 234  # More data is available.
ERROR_NO_MORE_ITEMS = 259  #  No more data is available.
ERROR_OPEN_FAILED = 110  # The system cannot open the device or file specified.
ERROR_BAD_QUERY_SYNTAX = 1615  # SQL query syntax invalid or unsupported.
ERROR_FUNCTION_FAILED = 1627  # Function failed during execution.
ERROR_CREATE_FAILED = 1631  # Data of this type is not supported.

def ff(lib, name, restype, *args):
    f = getattr(lib, name)
    if args:
        f.restype = restype
        if len(args) > 1:
            f.argtypes = list(args)
    else:
        f.restype = None
    return f

char_types = {'A':'A', 'W':'W', 'T':TCHAR_TYPE, '':''}

def add_foriegn_functions(lib, declarations, globals_):
    for declaration in declarations:
        name, char_type, restype = declaration[0:3]
        argtypes = declaration[3:]
        char_type = char_types[char_type]
        globals_[name] = ff(lib, name+char_type, restype, *argtypes)

msi_functions = [
    ('MsiOpenDatabase', 'T', c_uint, Tchar_p, Tchar_p, PHandle),
    ('MsiDatabaseOpenView', 'T', c_uint, Handle, Tchar_p, PHandle),
    ('MsiDatabaseCommit', '', c_uint, Handle),
    ('MsiViewClose', '', c_uint, Handle),
    ('MsiViewExecute', '', c_uint, Handle, Handle),
    ('MsiViewFetch', '', c_uint, Handle, PHandle),
    ('MsiViewGetColumnInfo', '', c_uint, Handle, MSICOLINFO, PHandle),
    ('MsiRecordDataSize', '', c_uint, Handle, c_uint),
    ('MsiRecordGetFieldCount', '', c_uint, Handle),
    ('MsiRecordGetInteger', '', c_int, Handle, c_uint),
    ('MsiRecordGetString', 'T', c_uint, Handle, c_uint, Tchar_p, P_ulong),
    ('MsiCreateRecord', '', Handle, c_uint),
    ('MsiRecordSetString', 'T', c_uint, Handle, c_uint, Tchar_p),
    ('MsiRecordSetStream', 'T', c_uint, Handle, c_uint, Tchar_p),
    ('MsiRecordReadStream', '', c_uint, Handle, c_uint,
                                POINTER(c_char), POINTER(c_ulong)),
    ('MsiRecordSetInteger', '', c_uint, Handle, c_uint, c_int),
    ('MsiCloseHandle', '', c_uint, Handle),
    ]

add_foriegn_functions(windll.Msi, msi_functions, globals())

class Record(object):
    class NullRecord(object):
        handle = 0
        
        def close(self):
            pass
    null_record = NullRecord()
    
    def __new__(cls, fields):
        if fields is None:
            return cls.null_record
        handle = MsiCreateRecord(len(fields))
        if handle == 0:
            raise InternalError("Unable to create a new record")
        try:
            for i, p in enumerate(fields):
                field = i + 1
                if isinstance(p, TStr):
                    rc = MsiRecordSetString(handle, field, p)
                elif isinstance(p, (int, long)):
                    rc = MsiRecordSetInteger(handle, field, p)
                elif isinstance(p, Stream):
                    rc = MsiRecordSetStream(handle, field, Stream.file_path)
                elif p is None:
                    rc = MsiRecordSetString(handle, field, NULL_TSTR)
                else:
                    raise ValueError("Unsupported parameter type %s"
                                     "for paramter %d" %
                                     (type(p), i))
                if rc != ERROR_SUCCESS:
                    raise error(rc)
            obj = object.__new__(cls)
            obj._handle = handle
            return obj
        except:
            MsiCloseHandle(handle)
            raise

    def close(self):
        if self._handle != None:
            MsiCloseHandle(self._handle)
            self._handle = None

    handle = property(lambda self: self._handle)

    def __del__(self):
        self.close()

def get_field_string(handle, field):
    size = MsiRecordDataSize(handle, field)
    if size == 0:
        return None
    buf = create_string_buffer(size+1)
    csize = c_ulong(len(buf))
    rc = MsiRecordGetString(handle, field, buf, byref(csize))
    if rc != ERROR_SUCCESS:
        raise error(rc)
    return buf[0:size]

def get_field_number(rec, field):
    value = MsiRecordGetInteger(rec, field)
    if value == MSI_NULL_INTEGER:
        return None
    return value

def get_field_stream(rec, field):
    size = MsiRecordDataSize(rec, field)
    buf = create_string_buffer(size)
    csize = c_ulong(size)
    rc = MsiRecordReadStream(rec, field, buf, byref(csize))
    if rc != ERROR_SUCCESS:
        raise error(rc)
    return buf

def column_info(view_handle, info_type):
    record_handle = Handle()
    rc = MsiViewGetColumnInfo(view_handle,
                              info_type,
                              byref(record_handle))
    if rc != ERROR_SUCCESS:
        raise error(rc)
    try:
        field_count = MsiRecordGetFieldCount(record_handle)
        return [get_field_string(record_handle, field)
                  for field in range(1, field_count+1)]
    finally:
        MsiCloseHandle(record_handle)

class Cursor(object):
    _select_pattern = re.compile(r"\s*select\s", flags=re.IGNORECASE)
    
    def __new__(cls, connection, key):
        obj = object.__new__(cls)
        obj._connection = connection
        obj._key = key
        obj._view = None
        obj._description = None
        obj._have_items = False
        obj._next_record = None
        return obj

    def execute(self, operation, parameters=None):
        if not self:
            raise OperationalError("Operation on a closed cursor")
        self._close_view()
        self._description = None
        self._have_items = False
        self._next_record = None
        chandle = Handle()
        rc = MsiDatabaseOpenView(self._connection._handle,
                                 operation,
                                 byref(chandle))
        if rc != ERROR_SUCCESS:
            raise error(rc)
        self._view = chandle.value
        try:
            record = Record(parameters)
            try:
                rc = MsiViewExecute(self._view, record.handle)
                if rc != ERROR_SUCCESS:
                    raise error(rc)
            finally:
                record.close()
            if self._select_pattern.match(operation) is None:
                self._close_view()
            else:
                definitions = column_info(self._view, MSICOLINFO_TYPES)
                if definitions:
                    names = column_info(self._view, MSICOLINFO_NAMES)
                    self._description = [
                        (nm,) + field_type(defn)
                           for nm, defn in itertools.izip(names,
                                                          definitions)]
                    self._get_next_record()
        except Exception:
            self._close_view()
            raise

    description = property(lambda self: self._description)

    def executemany(self, operation, seq_of_parameters):
        if not self:
            raise OperationalError("Operation on a closed cursor")
        self._close_view()
        self._description = None
        self._have_items = False
        self._next_record = None
        chandle = Handle()
        rc = MsiDatabaseOpenView(self._connection._handle,
                                 operation,
                                 byref(chandle))
        if rc != ERROR_SUCCESS:
            raise error(rc)
        self._view = chandle.value
        try:
            for parameters in seq_of_parameters:
                record = Record(parameters)
                try:
                    rc = MsiViewExecute(self._view, record.handle)
                    if rc != ERROR_SUCCESS:
                        raise error(rc)
                finally:
                    record.close()
                if (self._select_pattern.match(operation) is not None and
                    column_info(self._view, MSICOLINFO_TYPES)):
                    self._description = []
                    try:
                        self._get_next_record()
                    finally:
                        self._description = None
                    if self._have_items:
                        self._have_items = False
                        self._next_record = None
                        raise DatabaseError("Unexpected result set created by"
                                            " the operation")
        finally:
            self._close_view()

    def get_rowcount(self):
        if self._next_record is not None:
            return None
        if self._have_items:
            return 0
        return -1
    rowcount = property(get_rowcount)

    def fetchone(self):
        if not self._have_items:
            raise ProgrammingError("No records to fetch")
        record = self._next_record
        if record is not None:
            self._get_next_record()
        return record

    def fetchall(self):
        return [record for record in self]

    def close(self):
        if self:
            self._close_view()
            self._description = None
            self._next_record = None
            self._have_items = False
            self._connection._remove_cursor(self._key)
            self._connection = None
            self._status = 0

    def __nonzero__(self):
        return self._connection is not None

    def __del__(self):
        self.close()

    def _close_view(self):
        if self._view is not None:
            MsiViewClose(self._view)
            MsiCloseHandle(self._view)
            self._view = None

    def _get_next_record(self):
        self._next_record = None
        c_handle = Handle()
        rc = MsiViewFetch(self._view, byref(c_handle))
        if rc == ERROR_NO_MORE_ITEMS:
            self._close_view()
            return
        if rc != ERROR_SUCCESS:
            raise error(rc)
        self._have_items = True
        try:
            record = []
            for i, format in enumerate(self.description):
                typ = format[1]
                fld = i + 1
                if typ is OBJECT:
                    record.append(None)
                elif typ is STRING:
                    record.append(get_field_string(c_handle.value, fld))
                elif typ is NUMBER:
                    record.append(get_field_number(c_handle.value, fld))
                else:
                    record.append(get_field_stream(c_handle.value, fld))
            self._next_record = record
        finally:
            MsiCloseHandle(c_handle.value)

    def __iter__(self):
        return self

    def next(self):
        nxt = self.fetchone()
        if nxt is None:
            raise StopIteration()
        return nxt

class Connection(object):
    def __new__(cls, handle, commit_on_close):
        self = object.__new__(cls)
        self._handle = handle
        self._cursors = {}
        self._rollback = False
        self._commit_on_close = commit_on_close
        self._next_cursor_key = 0
        return self
        
    def close(self):
        if self:
            for cursor_weakref in self._cursors.values():
                cursor = cursor_weakref()
                if cursor is not None:
                    cursor.close()
            self._cursors = None
            if self._commit_on_close or not self._rollback:
                try:
                    self.commit()
                except StandardError:
                    pass
            MsiCloseHandle(self._handle)
            self._handle = None
            
    def __del__(self):
        self.close()
        
    def cursor(self):
        key = self._next_cursor_key
        cursor = Cursor(self, key)
        self._next_cursor_key += 1
        self._cursors[key] = weakref.ref(cursor)
        return cursor
    
    def commit(self):
        if not self:
            raise OperationalError("Commit not supported on a closed database")
        rc = MsiDatabaseCommit(self._handle)
        if rc != ERROR_SUCCESS:
            raise error(rc)

    def rollback(self):
        self._rollback = True

    def __nonzero__(self):
        return self._handle is not None

    def _remove_cursor(self, key):
        del self._cursors[key]

#=============================================================================
#   Public api

# database open read-only, no persistent changes
OPEN_READONLY = cast(0, Tchar_p)
# database read/write in transaction mode
OPEN_TRANSACT = cast(1, Tchar_p)
# database direct read/write without transaction
OPEN_DIRECT = cast(2, Tchar_p)
# create new database, direct mode read/write
OPEN_CREATE = cast(3, Tchar_p)

STRING = 'STRING'
NUMBER = 'NUMBER'
BINARY = 'BINARY'
OBJECT = 'OBJECT'

apilevel = '2.0'
threadsafety = 1
paramstyle = 'qmark'

class Error(StandardError):
    pass

class Warning(StandardError):
    pass

class InterfaceError(Error):
    pass

class DatabaseError(Error):
    pass

class InternalError(DatabaseError):
    pass

class OperationalError(DatabaseError):
    pass

class ProgrammingError(DatabaseError):
    pass

class IntegrityError(DatabaseError):
    pass

class DataError(DatabaseError):
    pass

class NotSupportedError(DatabaseError):
    pass

class Stream(object):
    def __init__(self, file_path):
        if not isinstance(file_path, TStr):
            raise TypeError("The file path is not of type %s" % TStr)
        self.file_path = file_path

def connect(dsn, persist=None):
    if persist is None:
        persist = OPEN_READONLY
    handle = Handle()
    rc = MsiOpenDatabase(dsn, persist, byref(handle))
    if rc != ERROR_SUCCESS:
        raise error(rc)
    try:
        return Connection(handle.value, persist is OPEN_DIRECT)
    except Exception:
        MsiCloseHandle(handle.value)

#=============================================================================
#   Api dependent stuff

msi_errors = {
    ERROR_INVALID_HANDLE:
      (InternalError, "The handle is invalid"),
    ERROR_BAD_QUERY_SYNTAX:
      (ProgrammingError, "SQL query syntax invalid or unsupported"),
    ERROR_FUNCTION_FAILED:
      (InternalError, "Function failed during execution"),
    ERROR_CREATE_FAILED:
      (ProgrammingError, "Data of this type is not supported"),
    ERROR_INVALID_PARAMETER:
      (ProgrammingError, "The parameter is incorrect"),
    ERROR_OPEN_FAILED:
      (OperationalError, "The system cannot open the device or file specified")
    }

def error(rc):
    try:
        typ, msg = msi_errors[rc]
    except IndexError:
        msg = "Unexpected MSI error %d" % rc
        typ = InternalError
    return typ(msg)

column_formats = {
    's': (STRING, False),
    'S': (STRING, True),
    'l': (STRING, False),
    'L': (STRING, True),
    'i': (NUMBER, False),
    'I': (NUMBER, True),
    'v': (BINARY, False),
    'V': (BINARY, True),
    'g': (STRING, False),
    'G': (STRING, True),
    'j': (NUMBER, False),
    'J': (NUMBER, True),
    'o': (OBJECT, True),
    'O': (OBJECT, True)
    }

number_specs = {
    0: (None, None, None),
    1: (4, 1, 3),
    2: (6, 2, 5),
    4: (11, 4, 10)
    }

def field_type(code):
    if code is None:
        return (None,) * 6
    data_type, null_ok = column_formats[code[0]]
    width = int(code[1:])
    if data_type is OBJECT:
        display_size = 0
        internal_size = None
        precision = None
    elif data_type is NUMBER:
        display_size, internal_size, precision = number_specs[width]
    elif width == 0:
        display_size = None
        internal_size = None
        precision = None
    else:
        display_size = width
        internal_size = width
        precision = None
    return (data_type, display_size, internal_size, precision, None, null_ok)

__all__ = ['OPEN_READONLY', 'OPEN_TRANSACT', 'OPEN_DIRECT', 'OPEN_CREATE',
           'STRING', 'NUMBER', 'BINARY', 'apilevel', 'threadsafety',
           'paramstyle','Error', 'Warning', 'InterfaceError', 'DatabaseError',
           'InternalError', 'OperationalError', 'ProgrammingError',
           'IntegrityError', 'DataError', 'NotSupportedError', 'Stream',
           'connect']
