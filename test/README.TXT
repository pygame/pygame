================================================================================
= README FOR PYGAME TESTS =
===========================

================================================================================
= run_tests.py =
================

The test runner for PyGame was developed for these purposes:
    
    * Per process isolation of test modules
    * Ability to tag tests for exclusion (interactive tests etc)
    * Record timings of tests

It does this by altering the behaviour of unittest at run time. As much as
possible each individual module was left to be fully compatible with the
standard unittest.

If an individual module is run, eg ``python test/color_test.py``, then it will
run an unmodified version of unittest. ( unittest.main() )

================================================================================
= Creating New Test Modules =
=============================

*** NOTE ***

Be sure to import test_utils first at the top of your file, this will set the
sys.path required for test.unittest to run, otherwise run_tests.py will not work
properly

import test_utils
import test.unittest as unittest

================================================================================
= Writing New Tests =
=====================

See test/util/gen_stubs.py for automatic creation of test stubs
Follow the naming convention

================================================================================
= gen_stubs.py =
================

trunk/test/util/gen_stubs.py

The gen_stubs.py utility will inspect pygame, and compile stubs of each of the
module's callables (funcs, methods, getter/setters). It will include in the
test's comment the __doc__ and the documentation found in the relevant xxxx.doc
files. There is a naming convention in place that maps test names to callables
in a one to one manner. If there are no untested (or unstubbed) callables then
gen_stubs.py will output nothing.

gen_stubs.py --help

gen_stubs.py module -d >> ../test_module_to_append_to.py

You will need to manually merge the stubs into relevant TestCases.

================================================================================
= Test Naming Convention =
==========================

This convention is in place so the stub generator can tell what has already been
tested and for other introspection purposes.

Each module in the pygame package has a corresponding test module in the
trunk/test directory.

    pygame.color : color_test.py

Each class has corresponding TestCase[s] in form of $Class + "Type"

    # TC[:TC.rindex('Type')]

    pygame.color.Color : color_test.ColorTypeTest
    pygame.color.Color : color_test.ColorTypeTestOtherAspect

    *** NOTE *** 

    Use the names of the instantiator helper functions:

        eg pygame.cdrom.CD and not pygame.cdrom.CDType

Each test should be named in the form, test_$funcname__$comment

    Surface.blit      : test_blit__raises_exception_if_locked

================================================================================
= Tagging =
===========

There are three levels of tagging available, module level, TestCase level and
individual test level.

For class and module level tagging assign a tag attribute __tags__ = []

Module Level Tags
-----------------

Include the module level tags in: 'some_module_tags.py'

Where the module name is 'some_module' which has its tests in some_module_test.py

This allows some modules to be excluded without loading some code in the first place.



# some_module_test.py
__tags__ = ['display', 'interactive']

Tags are inherited by children, so all TestCases, and thus tests will inherit
these module level tags.

Class Level Tags
----------------

If you want to override a specifig tag then you can use negation.

class SomeTest(unittest.TestCase):
    __tags__ = ['-interactive']

Test Level Tags
---------------

The tags for individual tests are specified in the __doc__ for the test.

format : |Tags:comma,separated,tags|

def test_something__about_something(self):
    """
    |Tags:interactive,some_other_tag|

    """



*** NOTE *** 

By default 'interactive' tags are not run

run_tests.py --exclude display,slow for exclusion of tags

However if you do python test/some_module_test.py all of the tests will run.

See run_tests.py --help for more details.



================================================================================
= test_utils.py =
=================

This contains utility routines for common testing needs as well as sets the
sys.path required for test.unittest to work.

some convenience functions:

question(q)
    Will ask q and return True if they answered yes

prompt(p)
    Will notify the user of p and then prompt them to "press enter to continue"

trunk_relative_path(pth)
    Will return a normalized relative path, relative to the test_module
    
    eg trunk_relative_path('examples\\data\\alien.jpg') will work on linux
    
    This is so the test module can be run from anywhere with working paths
        eg ../test/color_test.py 
        
fixture_path(pth)
    Likewise but paths are relative to trunk\test\fixtures

example_path(pth)
    Likewise but paths are relative to trunk\examples

================================================================================
