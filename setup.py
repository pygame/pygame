#!/usr/bin/env python

'''
'''

__docformat__ = 'restructuredtext'
__version__ = '$Id:$'

import os
from os.path import join, abspath, dirname, splitext
import sys
import subprocess

from distutils.core import setup
from distutils.cmd import Command


# A "do-everything" command class for building any type of documentation.
class BuildDocCommand(Command):
    user_options = [('doc-dir=', None, 'directory to build documentation'),
                    ('epydoc=', None, 'epydoc executable'),
                    ('rest-latex=', None, 'ReST LaTeX writer executable'),
                    ('rest-html=', None, 'ReST HTML writer executable'),
                    ('mkhowto=', None, 'Python mkhowto executable')]

    def initialize_options(self):
        self.doc_dir = join(abspath(dirname(sys.argv[0])), 'doc')
        self.epydoc = 'epydoc'
        self.rest_latex = 'python_latex.py'
        self.rest_html = 'rst2html.py'
        self.mkhowto = 'mkhowto'

    def finalize_options(self):
        pass

    def run(self):
        if 'pre' in self.doc:
            subprocess.call(self.doc['pre'], shell=True)

        prev_dir = os.getcwd()
        if 'chdir' in self.doc:
            dir = abspath(join(self.doc_dir, self.doc['chdir']))
            try:
                os.makedirs(dir)
            except:
                pass
            os.chdir(dir)

        if 'epydoc_packages' in self.doc:
            cmd = [self.epydoc,
                   '--no-private',
                   '--html',
                   '--no-sourcecode',
                   '--url=http://www.pygame.org',
                   '-v']
            cmd.append('--name="%s"' % self.doc['description'])
            if 'epydoc_dir' in self.doc:
                cmd.append('-o %s' % \
                    abspath(join(self.doc_dir, self.doc['epydoc_dir'])))
            cmd.append(self.doc['epydoc_packages'])
            subprocess.call(' '.join(cmd), shell=True)

        if 'rest_howto' in self.doc:
            txtfile = abspath(join(self.doc_dir, self.doc['rest_howto']))
            texfile = '%s.tex' % splitext(txtfile)[0]
            cmd = [self.rest_latex,
                   txtfile, 
                   '>',
                   texfile]
            subprocess.call(' '.join(cmd), shell=True)

            cmd = [self.mkhowto,
                   '--split=4',
                   texfile]
            subprocess.call(' '.join(cmd), shell=True)

        if 'rest_html' in self.doc:
            txtfile = abspath(join(self.doc_dir, self.doc['rest_html']))
            htmlfile = '%s.html' % splitext(txtfile)[0]
            cmd = [self.rest_html,
                   txtfile,
                   '>',
                   htmlfile]
            subprocess.call(' '.join(cmd), shell=True)

        os.chdir(prev_dir)

# Fudge a command class given a dictionary description
def make_doc_command(**kwargs):
    class c(BuildDocCommand):
        doc = dict(**kwargs)
        description = 'build %s' % doc['description']
    c.__name__ = 'build_doc_%s' % c.doc['name'].replace('-', '_')
    return c

# This command does nothing but run all the other doc commands.
# (sub_commands are set later)
class BuildAllDocCommand(Command):
    description = 'build all documentation'
    user_options = []
    sub_commands = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

extra_cmds = {
    'build_doc_sdl_api': make_doc_command(
         name='sdl-api',
         description='SDL-ctypes API documentation',
         chdir='.build-doc',
         pre='python support/prep_doc_sdl.py doc/.build-doc',
         epydoc_dir='sdl-api',
         epydoc_packages='SDL'),
    'build_doc_pygame_api': make_doc_command(
         name='pygame-api',
         description='Pygame-ctypes API',
         chdir='.build-doc',
         pre='python support/prep_doc_pygame.py doc/.build-doc',
         epydoc_dir='pygame-api',
         epydoc_packages='pygame'),
    'build_doc_sdl_manual': make_doc_command(
         name='sdl-manual',
         description='SDL-ctypes manual',
         chdir='',
         rest_howto='sdl-manual.txt'),
    'build_doc_index': make_doc_command(
         name='index',
         description='documentation index',
         chdir='',
         rest_html='index.txt'),
    'build_doc': BuildAllDocCommand
}

for name in extra_cmds.keys():
    if name != 'build_doc':
        BuildAllDocCommand.sub_commands.append((name, None))

setup(
    name='pygame-ctypes',
    version='0.06',
    description='Python game and multimedia package',
    author='Alex Holkner',
    author_email='aholkner@cs.rmit.edu.au',
    url='http://www.pygame.org/ctypes',
    license='LGPL',
    cmdclass=extra_cmds,
    packages=['SDL', 'pygame'],
)

