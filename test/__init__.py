"""Pygame unit test suite package

Exports function run()

A quick way to run the test suite package from the command line
is by importing the go submodule:

python -c "import pygame.tests.go" [<test options>]

Command line option --help displays a usage message.

The xxxx_test submodules of the tests package are unit test suites for
individual parts of Pygame. Each can also be run as a main program. This is
useful if the test, such as cdrom_test, is interactive.

For Pygame development the test suite can also be run from a Pygame installation
director. Installation directory program run_tests.py is provided for convenience,
though test/go.py and test/xxxx_test.py can be run directly.

"""

# Any tests in IGNORE will not be run
IGNORE = set ([
    "scrap_test",
])

# Subprocess has less of a need to worry about interference between tests
SUBPROCESS_IGNORE = set ([
    "scrap_test",
])

def run(my_name=None):
    """Run the Pygame unit test suite

    The run function is configured with command line options. The --help option
    displays a command line usage message. Optional argument my_name replaces
    the executable name in the help message.

    """
    
    if __name__ == 'pygame.tests':
        from pygame.tests.test_utils.run_tests import run
    else:
        from test.test_utils.run_tests import run
    run(my_name)
