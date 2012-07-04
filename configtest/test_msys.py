# program test_msys.py

"""Test msys.py.

This test requires that MSYS is installed and configured for MinGW."""

import sys
sys.path.append('..')

import msys

import unittest
import re
import os

has_drive = msys.has_drive

class FstabRegexTestCase(unittest.TestCase):

    def setUp(self):
        self.pattern = re.compile(msys.FSTAB_REGEX, re.MULTILINE)
        
    def test_firstline(self):
        """Ensure first line is checked"""
        fstab = ('c:/xxx /mingw\n'
                 'c:/foobar /msys\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'c:/xxx')

    def test_middleline(self):
        """Ensure a middle line is checked"""
        fstab = ('c:/xxx /msys\n'
                 'c:/foobar /mingw\n'
                 'c:/yyy /something\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'c:/foobar')

    def test_lastline(self):
        """Ensure the last line is checked"""
        fstab = ('c:/xxx /msys\n'
                 'c:/foobar /whatever\n'
                 'c:/yyy /mingw\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'c:/yyy')

    def test_notfound(self):
        """Ensure no match when /mingw is missing"""
        fstab = ('c:/xxx /msys\n'
                 'c:/foobar /whatever\n'
                 'c:/yyy /something\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is None)

    def test_extra(self):
        """Ensure no match for something like /mingwx"""
        fstab = 'c:/xxx /mingwx\n'
        ma = self.pattern.search(fstab)
        self.failUnless(ma is None)

    def test_extra_entries(self):
        """Ensure extra entries are allowed on the line"""
        fstab = 'c:/xxx /mingw x'
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'c:/xxx')

    def test_crlf(self):
        """Ensure \\r\\n line endings are handled"""
        fstab = ('c:/xxx /msys\r\n'
                 'c:/foobar /mingw\r\n'
                 'c:/yyy /something\r\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'c:/foobar')

    def test_leading_space(self):
        """Ensure leading space is ignored"""
        fstabs = [' c:/xxx /mingw\n', '  c:/xxx /mingw\n', '\tc:/xxx /mingw\n']
        for fstab in fstabs:
            ma = self.pattern.search(fstab)
            self.failUnless(ma is not None)
            self.failUnlessEqual(ma.groupdict()['path'], 'c:/xxx')

    def test_multiple_spaces(self):
        """Ensure multiple spaces are ingored"""
        fstabs = ['c:/foobar  /mingw\n', 'c:/foobar   /mingw\n', 'c:/foobar\t /mingw\n']
        for fstab in fstabs:
            ma = self.pattern.search(fstab)
            self.failUnless(ma is not None)
            self.failUnlessEqual(ma.groupdict()['path'], 'c:/foobar')

    def test_multi_element_path(self):
        """Ensure a multi-element path is recognized"""
        fstab = ('c:/xxx /msys\n'
                 'c:/foo/bar /mingw\n'
                 'c:/yyy /something\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'c:/foo/bar')

    def test_backslash(self):
        """Ensure the backslashes is recognized as a path separator"""
        fstab = ('c:\\xxx /msys\n'
                 'c:\\foobar /mingw\n'
                 'c:\\yyy /something\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'c:\\foobar')

    def test_upper_case(self):
        """Ensure upper case letters are accepted"""
        fstab = ('c:/xxx /msys\n'
                 'C:/FOOBAR /mingw\n'
                 'c:/yyy /something\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'C:/FOOBAR')

    def test_non_letters(self):
        """Ensure non-letter characters are accepted"""
        fstab = ('c:/xxx /msys\n'
                 'c:/-57.2(_)/s /mingw\n'
                 'c:/yyy /something')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], 'c:/-57.2(_)/s')

    def test_no_drive_letter(self):
        """Ensure a missing drive letter is accepted"""
        fstab = ('c:/xxx /msys\n'
                 '/foobar /mingw\n'
                 'c:/yyy /something\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is not None)
        self.failUnlessEqual(ma.groupdict()['path'], '/foobar')

    def test_relative_path(self):
        """Ensure a relative path is rejected"""
        fstab = ('c:/xxx /msys\n'
                 'c/foobar /mingw\n'
                 'c:/yyy /something\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is None)

    def test_invalid_characters(self):
        """Ensure invalid characters are rejected"""
        fstab = ('c:/*xxx /mingw\n'
                 'c:/?foobar /mingw\n'
                 'c:/%yyy /mingw\n'
                 '_:/%zzz /mingw\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is None)

    def test_drive_letters(self):
        """Ensure drive letters a to z are accepted"""
        for d in 'abcdefghijklmnopqrstuvwxyz':
            path = '%s:/xxx' % d
            fstab = '%s /mingw' % path
            ma = self.pattern.search(fstab)
            self.failUnless(ma is not None)
            self.failUnlessEqual(ma.groupdict()['path'], path)
            
    def test_upper_case_drive_letters(self):
        """Ensure drive letters A to Z are accepted"""
        for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            path = '%s:/xxx' % d
            fstab = '%s /mingw' % path
            ma = self.pattern.search(fstab)
            self.failUnless(ma is not None)
            self.failUnlessEqual(ma.groupdict()['path'], path)
            
    def test_doubled_separators(self):
        """Does the regular expression reject doubled path separators?"""
        fstab = ('c:/\\xxx /mingw\n'
                 'c://foobar /mingw\n'
                 'c:\\\\yyy /mingw\n'
                 'c:\\/zzz /mingw\n')
        ma = self.pattern.search(fstab)
        self.failUnless(ma is None)

    def test_root_directory(self):
        """Does the regular expression reject the root directory?"""
        # This just documents a quirk of the regular expression
        fstab = 'c:/ /mingw\n'
        ma = self.pattern.search(fstab)
        self.failUnless(ma is None)


class MsysToWindowsTestCase(unittest.TestCase):
    """Test Msys.msys_to_windows"""
    some_file_name = 'foo.txt'

    def setUp(self):
        self.msys = msys.Msys()

    def test_path_usr(self):
        """Ensure /usr translates"""
        self.failUnlessEqual(self.msys.msys_to_windows('/usr'),
                             self.msys.msys_root.replace(os.sep, '/'))

    def test_path_usr_somefile(self):
        """Ensure /usr/..... translates"""
        msys_path = '/usr/%s' % self.some_file_name
        win_path = os.path.join(self.msys.msys_root, self.some_file_name).replace(os.sep, '/')
        self.failUnlessEqual(self.msys.msys_to_windows(msys_path), win_path)

    def test_path_mingw(self):
        """Ensure /mingw translates"""
        self.failUnlessEqual(self.msys.msys_to_windows('/mingw'),
                             self.msys.mingw_root.replace(os.sep, '/'))

    def test_path_mingw_something(self):
        """Ensure /mingw/.... translates"""
        msys_path = '/mingw/%s' % self.some_file_name
        win_path = os.path.join(self.msys.mingw_root, self.some_file_name).replace(os.sep, '/')
        self.failUnlessEqual(self.msys.msys_to_windows(msys_path), win_path)

    def test_path_root(self):
        """Ensure / translates"""
        self.failUnlessEqual(self.msys.msys_to_windows('/'),
                             self.msys.msys_root.replace(os.sep, '/'))

    def test_path_root_something(self):
        """Ensure /.... translates"""
        msys_path = '/%s' % self.some_file_name
        win_path = os.path.join(self.msys.msys_root, self.some_file_name).replace(os.sep, '/')
        self.failUnlessEqual(self.msys.msys_to_windows(msys_path), win_path)

    def test_drive_letter_absolute(self):
        """Ensure x:/.... translates"""
        for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            msys_path = '/%s/%s' % (d, self.some_file_name)
            win_path = '%s:/%s' % (d, self.some_file_name)
            self.failUnlessEqual(self.msys.msys_to_windows(msys_path), win_path)

    def test_drive_letter_relative(self):
        """Ensure x:.... translates"""
        for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            msys_path = '%s:dir/%s' % (d, self.some_file_name)
            win_path = os.path.join('%s:' % d, 'dir', self.some_file_name).replace(os.sep, '/')
            self.failUnlessEqual(self.msys.msys_to_windows(msys_path), win_path)

    def test_path_relative(self):
        """Ensure relative paths translate"""
        msys_path = './dir/%s' % self.some_file_name
        win_path = os.path.join('.', 'dir', self.some_file_name).replace(os.sep, '/')
        self.failUnlessEqual(self.msys.msys_to_windows(msys_path), win_path)


class WindowsToMsysTestCase(unittest.TestCase):
    """Test Msys.windows_to_msys"""

    some_file_name = 'foo.txt'

    def setUp(self):
        self.msys = msys.Msys()

    def test_path_root(self):
        """Ensure MSYS directory maps to /usr"""
        win_path = os.path.join(self.msys.msys_root, self.some_file_name)
        msys_path = '/usr/%s' % self.some_file_name
        self.failUnlessEqual(self.msys.windows_to_msys(win_path), msys_path)

    def test_path_mingw(self):
        """Ensure MinGW directory maps to /mingw"""
        win_path = os.path.join(self.msys.mingw_root, self.some_file_name)
        msys_path = '/mingw/%s' % self.some_file_name
        self.failUnlessEqual(self.msys.windows_to_msys(win_path), msys_path)

    def test_drive_letter(self):
        """Ensure x:/.... translates"""
        for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            win_path = '%s:\%s' % (d, self.some_file_name)
            msys_path = '/%s/%s' % (d, self.some_file_name)
            self.failUnlessEqual(self.msys.windows_to_msys(win_path), msys_path)

    def test_foward_slashes(self):
        """Ensure forward slashes in a Windows path are recognized"""
        self.failUnlessEqual(self.msys.windows_to_msys('C:/one/two'), '/C/one/two')


class ShellTestCase(unittest.TestCase):

    def setUp(self):
        self.msys = msys.Msys()
        self.testscript_path = os.path.abspath('.\\testscript')
        open(self.testscript_path, 'wb').write('echo $XXXYYYZZZ\n')
        self.msys.environ['XXXYYYZZZ'] = 'passed'

    def tearDown(self):
        try:
            os.remove(self.testscript_path)
        except StandardError:
            pass
 
    def test_environment(self):
        """Ensure MINGW_ROOT_DIRECTORY is set"""
        self.failUnlessEqual(self.msys.environ['MINGW_ROOT_DIRECTORY'],
                             self.msys.mingw_root)

    def test_shell_script_return_value(self):
        """Ensure run_shell_script returns the return value of the shell"""
        self.failUnlessEqual(self.msys.run_shell_script('exit 0'), 0)
        self.failUnlessEqual(self.msys.run_shell_script('exit 42'), 42)

    def test_shell_script_environment(self):
        """Ensure environment variables are passed to the shell"""
        script = 'test x"$SOMETHING" == xsomevalue'
        self.msys.environ['SOMETHING'] = 'somevalue'
        working_dir = os.getcwd()
        self.failUnlessEqual(self.msys.run_shell_script(script), 0)
        self.failUnlessEqual(os.getcwd(), working_dir)

    def test_shell_command(self):
        """Ensure msys_shell_command works"""
        cmd = self.msys.windows_to_msys(self.testscript_path)
        working_dir = os.getcwd()
        self.failUnlessEqual(self.msys.run_shell_command([cmd]).strip(), 'passed')
        self.failUnlessEqual(os.getcwd(), working_dir)

if __name__ == '__main__':
    unittest.main()
