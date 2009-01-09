"""Run one or more Pygame unittest modules in the test directory

For command line options use the --help option.

"""

import test

import sys
import os

my_name = os.path.split(sys.argv[0])[1]
test.run(my_name)
