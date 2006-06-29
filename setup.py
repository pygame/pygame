#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id:$'

import os
import os.path
import sys
import subprocess

from distutils.core import setup
from distutils.cmd import Command

doc_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'doc')
apidoc_dir = os.path.join(doc_dir, 'api')

# customise me:
latex_writer = \
    '/usr/lib/python2.4/site-packages/docutils/tools/python_latex.py'
mkhowto = \
    '/usr/share/doc/Python-2.5a2/tools/mkhowto'
rst2html = \
    'rst2html.py'

class ApiDocCommand(Command):
    description = 'generate HTML API documentation'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.call('python support/prep_doc.py build_doc/',
                        shell=True)
        try:
            os.makedirs(apidoc_dir)
        except:
            pass
        olddir = os.getcwd()
        os.chdir('build_doc')
        subprocess.call('epydoc --no-private --html --no-sourcecode ' + \
                        '--name=SDL-ctypes --url=http://www.pygame.org ' + \
                        '-v -o %s SDL' % apidoc_dir,
                        shell=True)
        os.chdir(olddir)

class ManualDocCommand(Command):
    description = 'generate HTML manual documentation'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        olddir = os.getcwd()
        os.chdir(doc_dir)
        subprocess.call('%s manual.txt > manual.tex' % latex_writer, shell=True)
        subprocess.call('%s --split=4 manual.tex' % mkhowto, shell=True)
        os.chdir(olddir)

class DocCommand(Command):
    description = 'generate HTML index page'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        olddir = os.getcwd()
        os.chdir(doc_dir)
        subprocess.call('%s index.txt > index.html' % rst2html, shell=True)
        os.chdir(olddir)

setup(
    name='SDL-ctypes',
    version='0.04',
    description='ctypes wrapper for SDL',
    author='Alex Holkner',
    author_email='aholkner@cs.rmit.edu.au',
    url='http://www.pygame.org/ctypes',
    license='LGPL',
    cmdclass={'apidoc':ApiDocCommand, 
              'manual':ManualDocCommand,
              'doc':DocCommand},
    packages=['SDL'],
)

