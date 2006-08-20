#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id$'

from setuptools import setup, find_packages

setup(
    name='Pygame-ctypes',
    version='0.09',
    packages=['pygame'],
    install_requires=['SDL-ctypes>=0.09'],
    package_data = {
        'pygame': ['*.bmp', '*.ttf']
    },
    zip_safe=True,
    description='Python game and multimedia package',
    author='Alex Holkner',
    author_email='aholkner@cs.rmit.edu.au',
    url='http://www.pygame.org/ctypes',
    license='LGPL')
    
