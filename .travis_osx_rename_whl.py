"""Rename the wheel so that pip will find it for more Pythons.

Python.org builds and Mac system Python are built as 'fat' binaries, including
both x86 (32 bit) and x86_64 (64 bit) executable code in one file. On these
Pythons, pip will look for wheels with fat libraries, tagged 'intel'. However,
all recent Mac systems run the 64 bit code.

Therefore, this script tells a small lie about the wheels we're producing. By
claiming they are fat ('intel') wheels, pip will install them on more Python
installations. This should not cause problems for the vast majority of users.
"""
import glob
import os
import sys

# There should be exactly one .whl
filenames = glob.glob('dist/*.whl')

if len(filenames) < 1:
    sys.exit("No wheels found")
elif len(filenames) > 1:
    print("Multiple wheels found:")
    for f in filenames:
        print("  {}".format(f))
    sys.exit(1)

path = filenames[0]

if '_intel' in path:
    print("Wheel already tagged 'intel':")
    print(path)
    sys.exit(0)
elif '_x86_64' not in path:
    print("Didn't find '_x86_64' in wheel filename:")
    print(path)
    sys.exit(1)

new_path = path.replace('_x86_64', '_intel')
os.rename(path, new_path)
print("Renamed wheel to:")
print(new_path)
