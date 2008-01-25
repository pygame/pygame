# test_config_msys_i.py program

"""Unit test of config_msys.py internals

This test need not be run from an MSYS console.
"""

import os
import sys

msys_root_directory = 'C:/_msys_/1.0'
mingw_root_directory = 'C:/_mingw_'
os.environ['SHELL'] = os.path.join(msys_root_directory, 'bin', 'sh.exe')
os.environ['MINGW_ROOT_DIRECTORY'] = mingw_root_directory

sys.path.append('..')

import config_msys

import unittest

def join_path(b, *p):
    return os.path.join(b, *p).replace(os.sep, '/')

class PathsTestCase(unittest.TestCase):
    """Test congif_msys.msys_to_windows"""
    some_file_name = 'foo.txt'

    def test_path_usr(self):
        """Ensure /usr translates"""
        self.failUnlessEqual(config_msys.msys_to_windows('/usr'), msys_root_directory)

    def test_path_usr_somefile(self):
        """Ensure /usr/..... translates"""
        msys_path = '/usr/%s' % self.some_file_name
        win_path = join_path(msys_root_directory, self.some_file_name)
        self.failUnlessEqual(config_msys.msys_to_windows(msys_path), win_path)

    def test_path_mingw(self):
        """Ensure /mingw translates"""
        self.failUnlessEqual(config_msys.msys_to_windows('/mingw'), mingw_root_directory)

    def test_path_mingw_something(self):
        """Ensure /mingw/.... translates"""
        msys_path = '/mingw/%s' % self.some_file_name
        win_path = join_path(mingw_root_directory, self.some_file_name)
        self.failUnlessEqual(config_msys.msys_to_windows(msys_path), win_path)

    def test_path_root(self):
        """Ensure / translates"""
        self.failUnlessEqual(config_msys.msys_to_windows('/'), msys_root_directory)

    def test_path_root_something(self):
        """Ensure /.... translates"""
        msys_path = '/%s' % self.some_file_name
        win_path = join_path(msys_root_directory, self.some_file_name)
        self.failUnlessEqual(config_msys.msys_to_windows(msys_path), win_path)

    def test_drive_letter_absolute(self):
        """Ensure x:/.... translates"""
        for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            msys_path = '/%s/%s' % (d, self.some_file_name)
            win_path = '%s:/%s' % (d, self.some_file_name)
            self.failUnlessEqual(config_msys.msys_to_windows(msys_path), win_path)

    def test_drive_letter_relative(self):
        """Ensure x:.... translates"""
        for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            msys_path = '%s:dir/%s' % (d, self.some_file_name)
            win_path = join_path('%s:' % d, 'dir', self.some_file_name)
            self.failUnlessEqual(config_msys.msys_to_windows(msys_path), win_path)

    def test_path_relative(self):
        """Ensure relative paths translate"""
        msys_path = './dir/%s' % self.some_file_name
        win_path = join_path('.', 'dir', self.some_file_name)
        self.failUnlessEqual(config_msys.msys_to_windows(msys_path), win_path)

if __name__ == '__main__':
    unittest.main()
