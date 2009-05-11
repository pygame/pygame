"""Pygame unit test suite package

Exports function run()

A quick way to run the test suite package from the command line
is by importing the go submodule:

python -c "import pygame.tests.go" [&lt;test options&gt;]

Command line option --help displays a usage message. Available options
correspond to the pygame.tests.run arguments.

The xxxx_test submodules of the tests package are unit test suites for
individual parts of Pygame. Each can also be run as a main program. This is
useful if the test, such as cdrom_test, is interactive.

For Pygame development the test suite can be run from a Pygame 
distribution root directory. Program run_tests.py is provided for convenience,
though test/go.py can be run directly.

"""

if __name__ == 'pygame.tests':
    from pygame.tests.test_utils.run_tests import run
else:
    from test.test_utils.run_tests import run

