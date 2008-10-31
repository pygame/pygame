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
= Writing New Tests =
=====================

See test/util/gen_stubs.py for automatic creation of test stubs
Follow the naming convention

================================================================================
= gen_stubs.py =
================

/test/util/gen_stubs.py

The gen_stubs.py utility will inspect pygame, and compile stubs of each of the
module's callables (funcs, methods, getter/setters). It will include in the
test's comment the __doc__ and the documentation found in the relevant xxxx.doc
files. There is a naming convention in place that maps test names to callables
in a one to one manner. If there are no untested (or unstubbed) callables then
gen_stubs.py will output nothing.

python gen_stubs.py --help

python gen_stubs.py module >> ../test_module_to_append_to.py

You will need to manually merge the stubs into relevant TestCases.

================================================================================
= Test Naming Convention =
==========================

This convention is in place so the stub generator can tell what has already been
tested and for other introspection purposes.

Each module in the pygame package has a corresponding test module in the
test/ directory.

    pygame2.base.color : base_color_test.py

Each test should be named in the form, test_$funcname__$comment

    Color.normalize      : test_Color_normalize__additional_note

================================================================================
= Tagging =
===========

There are three levels of tagging available, module level, TestCase level and
individual test level.

For class and module level tagging assign a tag attribute __tags__ = []

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

python run_tests.py --exclude display,slow for exclusion of tags

================================================================================
