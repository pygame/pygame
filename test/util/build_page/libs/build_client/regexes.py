################################################################################

import re

def Regex(string, test, flags=None):
    if flags: regex = re.compile(string, flags)
    else:     regex = re.compile(string)
    assert regex.search(test)
    return regex

################################################################################
# =================================
# = UNIT TEST REGULAR EXPRESSIONS =
# =================================

UNITTEST_ERRORS_TEST = r"""
loading event_test
.......F...
======================================================================
FAIL: EventModuleTest.test_set_blocked
----------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\PyGame\trunk\test\event_test.py", line 65, in test_set_blocked
    self.assertEqual(should_be_blocked, [])
AssertionError: [<Event(2-KeyDown {})>] != []

----------------------------------------------------------------------
Ran 11 tests in 0.511s

FAILED (failures=1)
"""

################################################################################
# ERROR_MATCHES_RE
#

# re.VERBOSE won't work with this particular expression

string = (
    r'^'
    r'(?P<test>(?:ERROR|FAIL): [^\n]+)'
    r'\n+-+\n+'
    r'(?P<traceback>(?:[^\n]+\n)+)'
    r'\n'
)

ERROR_MATCHES_RE = Regex(string, UNITTEST_ERRORS_TEST, re.M)

################################################################################
# TRACEBACK_RE

string = r'File "(?P<file>[^"]+)", line (?P<line>[0-9]+)'

TRACEBACK_RE = Regex(string, UNITTEST_ERRORS_TEST, 0)

################################################################################
# TESTS_FAILED_RE

TESTS_FAILED_RE = Regex ( "FAILED \([^\)]+=([0-9]+)\)", UNITTEST_ERRORS_TEST)

################################################################################
# ==================================
# = SUBVERSION REGULAR EXPRESSIONS =
# ==================================

SVN_BLAME_TEST = r"""
  1440    akalias             shutil.move(installer_dist_path, installer_path)
  1440    akalias             
  1440    akalias             temp_install_path = os.path.join(os.getcwd(), "install_test")
  1440    akalias             if os.path.exists(temp_install_path):
  1440    akalias                 cleardir(temp_install_path)
"""

################################################################################
# SVN_BLAME_RE

string = (
    r'\s*'
    r'(?P<revision>[0-9]+)'
    r'\s+'
    r'(?P<user>[^ ]+)'
    r'\s'
    r'(?P<blame_line>[^\r\n]*)'
)

SVN_BLAME_RE = Regex(string, SVN_BLAME_TEST)

################################################################################
# ================================
# = SETUP.PY REGULAR EXPRESSIONS =
# ================================

SETUP_PY_ERROR_TEST = r"""
src/font.c:663: error: initializer element is not constant
src/font.c:663: error: (near initialization for `font_builtins[3].ml_doc')
src/font.c:663: error: initializer element is not constant
src/font.c:663: error: (near initialization for `font_builtins[3]')
src/font.c:665: error: `DOC_PYGAMEFONTGETDEFAULTFONT' undeclared here (not in a function)
src/font.c:665: error: initializer element is not constant
src/font.c:665: error: (near initialization for `font_builtins[4].ml_doc')
src/font.c:665: error: initializer element is not constant
src/font.c:665: error: (near initialization for `font_builtins[4]')
src/font.c:666: error: initializer element is not constant
src/font.c:666: error: (near initialization for `font_builtins[5]')
"""

SETUP_PY_WARNING_TEST = r"""
c:/msys/1.0/local/include/SDL/SDL_main.h:50:1: warning: "main" redefined
src/surface.c: In function `surf_set_at':
src/surface.c:582: warning: unused variable `intobj'
src/surface.c: In function `surf_get_bounding_rect':
src/surface.c:1828: warning: suggest parentheses around && within ||
src/surface.c:1848: warning: suggest parentheses around && within ||
src/surface.c:1869: warning: suggest parentheses around && within ||
src/surface.c:1889: warning: suggest parentheses around && within ||
"""

################################################################################
# BUILD_WARNINGS_RE && BUILD_ERRORS_RE

string = (
    r"^(?P<file>[^\(\s]+\.c)"
    r"(?:\(|:)"
    r"(?P<line>[0-9]+)"
    r"(?:\)|:) ?:? %s:? "
    r"(?P<message>[^\r\n]+)"
    r"[\r\n]"
)

BUILD_WARNINGS_RE = Regex(string % 'warning',SETUP_PY_WARNING_TEST, re.MULTILINE)
BUILD_ERRORS_RE = Regex(string % 'error', SETUP_PY_ERROR_TEST, re.MULTILINE)

################################################################################
# LINK_ERRORS_RE

string = (
    r"^(?P<source_name>[^\(\s]+)"
    r"\.obj : error "
    r"(?P<message>[^\r\n]+)"
    r"[\r\n]"
)

# TODO: TEST 
LINK_ERRORS_RE = re.compile(string, re.MULTILINE)

################################################################################
# BUILD_TRACEBACK_RE

string = (
    r"^Traceback \\"
    r"(?P<traceback>most recent call [a-z]+\\)"
    r":[\r\n]+"
    r"(?P<message>.+[^\r\n]+Error:[^\r\n]+)"
)

# TODO: TEST
BUILD_TRACEBACK_RE = re.compile(string, re.MULTILINE | re.DOTALL)

################################################################################

__all__ = [a for a in dir() if a.endswith("_RE")]

################################################################################

if __name__ == '__main__':
    for attr in __all__:
        print("%s," % attr)

################################################################################
