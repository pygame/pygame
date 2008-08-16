================================================================================
= README FOR PYGAME TESTS =
===========================

================================================================================
= run_tests.py =
================

The test runner for PyGame was developed for these purposes:
    
    * Isolation of test modules per process
    * Tagging of tests for exclusion by tag (interactive tests etc)
    * Timing of tests

It does this by altering the behaviour of unittest at run time. As much as
possible each individual module was left to be fully compatible with the
standard unittest. At one point it was discovered that the run time patching of
unittest was incompatible with the 2.4 version so a copy of the 2.5 version now
is included in the pygame test  directory. The test directory was made a
package.

If an individual module is run, eg ``python test/color_test.py``, then it will
run an unmodified version of unittest. ( unittest.main() )

================================================================================
= Creating New Test Modules =
=============================

*** NOTE ***

Be sure to import test_utils first at the top of your file, this will set the
sys.path required for test.unittest to run, otherwise run_tests.py will work
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

The gen_stubs.py utility will inspect pygame, and compile stubs of the each
module's callables (funcs, methods, getter/setters). It will include in the
test's comment the __doc__ and the documentation found in the relevant xxxx.doc
files. There is a naming convention in place that maps test names to callables
in a one to one mapping. If there are no untested (or unstubbed) callables then
gen_stubs.py will output nothing.

gen_stubs.py --help

gen_stubs.py module -d >> ../test_module_to_append_to.py

You will need to manually merge the stubs into relevant TestCases.

================================================================================
= Test Naming Convention =
==========================

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

Each test should be name in the form,   test_$funcname__comment

    Surface.blit      : test_blit__raises_exception_if_locked

================================================================================
= Tagging =
===========

There are three levels of tagging available, module level, TestCase level and
individual test level.

For class and module level tagging, assign a tag attribute __tags__ = []

Module Level Tags
-----------------

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

================================================================================
= test_utils.py =
=================

This contains utility routines for common testing needs as well as sets the
sys.path required for test.unittest to work.

================================================================================