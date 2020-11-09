#!/usr/bin/env python
#
# This is the distutils setup script for pygame.
# Full instructions are in https://www.pygame.org/wiki/GettingStarted
#
# To configure, compile, install, just run this script.
#     python setup.py install

import io

with io.open('README.rst', encoding='utf-8') as readme:
    LONG_DESCRIPTION = readme.read()

EXTRAS = {}

METADATA = {
    "name":             "pygame",
    "version":          "2.0.1.dev1",
    "license":          "LGPL",
    "url":              "https://www.pygame.org",
    "author":           "A community project.",
    "author_email":     "pygame@pygame.org",
    "description":      "Python Game Development",
    "long_description": LONG_DESCRIPTION,
}

import re
import sys
import os

# just import these always and fail early if not present
import distutils
from setuptools import setup

IS_PYPY = '__pypy__' in sys.builtin_module_names

def compilation_help():
    """ On failure point people to a web page for help.
    """
    import platform
    the_system = platform.system()
    if the_system == 'Linux':
        if hasattr(platform, 'linux_distribution'):
            distro = platform.linux_distribution()
            if distro[0].lower() == 'ubuntu':
                the_system = 'Ubuntu'
            elif distro[0].lower() == 'debian':
                the_system = 'Debian'

    help_urls = {
        'Linux': 'https://www.pygame.org/wiki/Compilation',
        'Ubuntu': 'https://www.pygame.org/wiki/CompileUbuntu',
        'Debian': 'https://www.pygame.org/wiki/CompileDebian',
        'Windows': 'https://www.pygame.org/wiki/CompileWindows',
        'Darwin': 'https://www.pygame.org/wiki/MacCompile',
    }

    default = 'https://www.pygame.org/wiki/Compilation'
    url = help_urls.get(the_system, default)

    if IS_PYPY:
        url += '\n    https://www.pygame.org/wiki/CompilePyPy'

    print ('\n---')
    print ('For help with compilation see:')
    print ('    %s' % url)
    print ('To contribute to pygame development see:')
    print ('    https://www.pygame.org/contribute.html')
    print ('---\n')



if not hasattr(sys, 'version_info') or sys.version_info < (2,7):
    compilation_help()
    raise SystemExit("Pygame requires Python version 2.7 or above.")
if sys.version_info >= (3, 0) and sys.version_info < (3, 4):
    compilation_help()
    raise SystemExit("Pygame requires Python3 version 3.5 or above.")
if IS_PYPY and sys.pypy_version_info < (7,):
    raise SystemExit("Pygame requires PyPy version 7.0.0 above, compatible with CPython 2.7 or CPython 3.5+")

def consume_arg(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False

#get us to the correct directory
path = os.path.split(os.path.abspath(sys.argv[0]))[0]
os.chdir(path)
#os.environ["CFLAGS"] = "-W -Wall -Wpointer-arith -Wcast-qual -Winline " + \
#                       "-Wcast-align -Wconversion -Wstrict-prototypes " + \
#                       "-Wmissing-prototypes -Wmissing-declarations " + \
#                       "-Wnested-externs -Wshadow -Wredundant-decls"
if consume_arg("-warnings"):
    os.environ["CFLAGS"] = "-W -Wimplicit-int " + \
                       "-Wimplicit-function-declaration " + \
                       "-Wimplicit -Wmain -Wreturn-type -Wunused -Wswitch " + \
                       "-Wcomment -Wtrigraphs -Wformat -Wchar-subscripts " + \
                       "-Wuninitialized -Wparentheses " +\
                       "-Wpointer-arith -Wcast-qual -Winline -Wcast-align " + \
                       "-Wconversion -Wstrict-prototypes " + \
                       "-Wmissing-prototypes -Wmissing-declarations " + \
                       "-Wnested-externs -Wshadow -Wredundant-decls"

if consume_arg('-pygame-ci'):
    cflags = os.environ.get('CFLAGS', '')
    if cflags:
        cflags += ' '
    cflags += '-Werror=nested-externs -Werror=switch -Werror=implicit ' + \
              '-Werror=implicit-function-declaration -Werror=return-type ' + \
              '-Werror=implicit-int -Werror=main -Werror=pointer-arith ' + \
              '-Werror=format-security -Werror=uninitialized ' + \
              '-Werror=trigraphs -Werror=parentheses -Werror=unused-value ' + \
              '-Werror=cast-align -Werror=int-conversion ' + \
              '-Werror=incompatible-pointer-types'
    os.environ['CFLAGS'] = cflags

# For python 2 we remove the -j options.
if sys.version_info[0] < 3:
    # Used for parallel builds with setuptools. Not supported by py2.
    [consume_arg('-j%s' % x) for x in range(32)]


STRIPPED=False

# STRIPPED builds don't have developer resources like docs or tests

if "PYGAME_ANDROID" in os.environ:
    # test cases and docs are useless inside an APK
    STRIPPED=True

if consume_arg('-stripped'):
    STRIPPED=True

enable_arm_neon = False
if consume_arg('-enable-arm-neon'):
    enable_arm_neon = True
    cflags = os.environ.get('CFLAGS', '')
    if cflags:
        cflags += ' '
    cflags += '-mfpu=neon'
    os.environ['CFLAGS'] = cflags

if consume_arg('cython'):
    # compile .pyx files
    # So you can `setup.py cython` or `setup.py cython install`
    try:
        from Cython.Build.Dependencies import cythonize_one
    except ImportError:
        print("You need cython. https://cython.org/, pip install cython --user")
        sys.exit(1)

    from Cython.Build.Dependencies import create_extension_list
    from Cython.Build.Dependencies import create_dependency_tree

    try:
        from Cython.Compiler.Main import Context
        from Cython.Compiler.Options import CompilationOptions, default_options

        c_options = CompilationOptions(default_options)
        ctx = Context.from_options(c_options)
    except ImportError:
        from Cython.Compiler.Main import Context, CompilationOptions, default_options

        c_options = CompilationOptions(default_options)
        ctx = c_options.create_context()

    import glob
    pyx_files = glob.glob(os.path.join('src_c', 'cython', 'pygame', '*.pyx')) + \
                glob.glob(os.path.join('src_c', 'cython', 'pygame', '**', '*.pyx'))

    pyx_files, pyx_meta = create_extension_list(pyx_files, ctx=ctx)
    deps = create_dependency_tree(ctx)

    queue = []

    for ext in pyx_files:
        pyx_file = ext.sources[0] # TODO: check all sources, extension

        c_file = os.path.splitext(pyx_file)[0].split(os.path.sep)
        del c_file[1:3] # output in src_c/
        c_file = os.path.sep.join(c_file) + '.c'

        # update outdated .c files
        if os.path.isfile(c_file):
            c_timestamp = os.path.getmtime(c_file)
            if c_timestamp < deps.timestamp(pyx_file):
                dep_timestamp, dep = deps.timestamp(pyx_file), pyx_file
                priority = 0
            else:
                dep_timestamp, dep = deps.newest_dependency(pyx_file)
                priority = 2 - (dep in deps.immediate_dependencies(pyx_file))
            if dep_timestamp > c_timestamp:
                outdated = True
            else:
                outdated = False
        else:
            outdated = True
            priority = 0
        if outdated:
            print('Compiling {} because it changed.'.format(pyx_file))
            queue.append((priority, dict( pyx_file=pyx_file, c_file=c_file, fingerprint=None, quiet=False,
                                          options=c_options, full_module_name=ext.name,
                                          embedded_metadata=pyx_meta.get(ext.name) )))

    # compile in right order
    queue.sort(key=lambda a: a[0])
    queue = [pair[1] for pair in queue]

    count = len(queue)
    for i, kwargs in enumerate(queue):
        kwargs['progress'] = '[{}/{}] '.format(i + 1, count)
        cythonize_one(**kwargs)


AUTO_CONFIG = False
if consume_arg('-auto'):
    AUTO_CONFIG = True

import os.path, glob, stat, shutil
import distutils.sysconfig
from distutils.core import setup, Command
from distutils.extension import read_setup_file
from distutils.command.install_data import install_data
from distutils.command.sdist import sdist


revision = ''

def add_datafiles(data_files, dest_dir, pattern):
    """Add directory structures to data files according to a pattern"""
    src_dir, elements = pattern
    def do_directory(root_dest_path, root_src_path, elements):
        files = []
        for e in elements:
            if isinstance(e, list):
                src_dir, elems = e
                dest_path = '/'.join([root_dest_path, src_dir])
                src_path = os.path.join(root_src_path, src_dir)
                do_directory(dest_path, src_path, elems)
            else:
                files.extend(glob.glob(os.path.join(root_src_path, e)))
        if files:
            data_files.append((root_dest_path, files))
    do_directory(dest_dir, src_dir, elements)

# # allow optionally using setuptools for bdist_egg.
# if consume_arg("-setuptools") in sys.argv:
#     from setuptools import setup
#     sys.argv.remove ("-setuptools")

# we need to eat this argument in to distutils doesn't trip over it
consume_arg("-setuptools")


# NOTE: the bdist_mpkg_support is for darwin.
try:
    import bdist_mpkg_support
except ImportError:
    pass
else:
    EXTRAS.update({
        'options': bdist_mpkg_support.options,
        'setup_requires': ['bdist_mpkg>=0.4.2'],
        #'install_requires': ['pyobjc'],
        #'dependency_links': ['http://rene.f0o.com/~rene/stuff/macosx/']
    })

#headers to install
headers = glob.glob(os.path.join('src_c', '*.h'))
headers.remove(os.path.join('src_c', 'scale.h'))
headers.append(os.path.join('src_c', 'include'))

import distutils.command.install_headers

# monkey patch distutils header install to copy over directories
def run_install_headers(self):
    headers = self.distribution.headers
    if not headers:
        return

    self.mkpath(self.install_dir)
    for header in headers:
        if os.path.isdir(header):
            destdir=os.path.join(self.install_dir, os.path.basename(header))
            self.mkpath(destdir)
            for entry in os.listdir(header):
                header1=os.path.join(header, entry)
                if not os.path.isdir(header1):
                    (out, _) = self.copy_file(header1, destdir)
                    self.outfiles.append(out)
        else:
            (out, _) = self.copy_file(header, self.install_dir)
            self.outfiles.append(out)

distutils.command.install_headers.install_headers.run = run_install_headers

# option for not installing the headers.
if consume_arg("-noheaders"):
    headers = []

#sanity check for any arguments
if len(sys.argv) == 1 and sys.stdout.isatty():
    if sys.version_info[0] >= 3:
        reply = input('\nNo Arguments Given, Perform Default Install? [Y/n]')
    else:
        reply = raw_input('\nNo Arguments Given, Perform Default Install? [Y/n]')
    if not reply or reply[0].lower() != 'n':
        sys.argv.append('install')


#make sure there is a Setup file
if AUTO_CONFIG or not os.path.isfile('Setup'):
    print ('\n\nWARNING, No "Setup" File Exists, Running "buildconfig/config.py"')
    import buildconfig.config
    try:
        buildconfig.config.main(AUTO_CONFIG)
    except:
        compilation_help()
        raise
    if '-config' in sys.argv:
        sys.exit(0)
    print ('\nContinuing With "setup.py"')


try:
    s_mtime = os.stat("Setup")[stat.ST_MTIME]
    sin_mtime = os.stat(os.path.join('buildconfig', 'Setup.SDL1.in'))[stat.ST_MTIME]
    if sin_mtime > s_mtime:
        print ('\n\nWARNING, "buildconfig/Setup.SDL1.in" newer than "Setup",'
               'you might need to modify "Setup".')
except OSError:
    pass

# get compile info for all extensions
try:
    extensions = read_setup_file('Setup')
except:
    print ("""Error with the "Setup" file,
perhaps make a clean copy from "Setup.in".""")
    compilation_help()
    raise

# Only define the ARM_NEON defines if they have been enabled at build time.
if enable_arm_neon:
    for e in extensions:
        e.define_macros.append(('PG_ENABLE_ARM_NEON', '1'))

# if not building font, try replacing with ftfont
alternate_font = os.path.join('src_py', 'font.py')
if os.path.exists(alternate_font):
    os.remove(alternate_font)

have_font = False
have_freetype = False
for e in extensions:
    if e.name == 'font':
        have_font = True
    if e.name == '_freetype':
        have_freetype = True
if not have_font and have_freetype:
    shutil.copyfile(os.path.join('src_py', 'ftfont.py'), alternate_font)

#extra files to install
data_path = os.path.join(distutils.sysconfig.get_python_lib(), 'pygame')
pygame_data_files = []
data_files = [('pygame', pygame_data_files)]

#add files in distribution directory
# pygame_data_files.append('LGPL')
# pygame_data_files.append('readme.html')
# pygame_data_files.append('install.html')

add_stubs = True
# add *.pyi files into distribution directory
if add_stubs:
    pygame_data_files.append(os.path.join('buildconfig', 'pygame-stubs', 'py.typed'))
    type_files = glob.glob(os.path.join('buildconfig', 'pygame-stubs', '*.pyi'))
    for type_file in type_files:
        pygame_data_files.append(type_file)
    _sdl2 = glob.glob(os.path.join('buildconfig', 'pygame-stubs', '_sdl2', '*.pyi'))
    if _sdl2:
        _sdl2_data_files = []
        data_files.append(('pygame/_sdl2', _sdl2_data_files))
        for type_file in _sdl2:
            _sdl2_data_files.append(type_file)


#add non .py files in lib directory
for f in glob.glob(os.path.join('src_py', '*')):
    if not f[-3:] == '.py' and not f[-4:] == '.doc' and os.path.isfile(f):
        pygame_data_files.append(f)

# We don't need to deploy tests, example code, or docs inside a game

#tests/fixtures
add_datafiles(data_files, 'pygame/tests',
              ['test',
                  [['fixtures',
                      [['xbm_cursors',
                          ['*.xbm']],
                       ['fonts',
                          ['*.ttf', '*.otf', '*.bdf', '*.png']]]]]])

#examples
add_datafiles(data_files, 'pygame/examples',
              ['examples', ['README.rst', ['data', ['*']]]])

#docs
add_datafiles(data_files, 'pygame/docs',
              ['docs',
                  ['*.html',             # Navigation and help pages
                   '*.gif',              # pygame logos
                   '*.js',               # For doc search
                   ['ref',               # pygame reference
                       ['*.html',        # Reference pages
                        '*.js',          # Comments script
                        '*.json']],      # Comment data
                   ['c_api',             # pygame C API
                       ['*.html']],
                   ['tut',               # Tutorials
                       ['*.html',
                        ['tom',
                            ['*.html',
                             '*.png']]]],
                   ['_static',            # Sphinx added support files
                        ['*.css',
                         '*.png',
                         '*.ico',
                         '*.js']],
                   ['_images',            # Sphinx added reST ".. image::" refs
                        ['*.jpg',
                         '*.png',
                         '*.gif']],
                   ['_sources',           # Used for ref search
                        ['*.txt',
                         ['ref',
                            ['*.txt']]]]]])

#generate the version module
def parse_version(ver):
    return ', '.join(s for s in re.findall(r'\d+', ver)[0:3])

def parse_source_version():
    pgh_major = -1
    pgh_minor = -1
    pgh_patch = -1
    major_exp_search = re.compile(r'define\s+PG_MAJOR_VERSION\s+([0-9]+)').search
    minor_exp_search = re.compile(r'define\s+PG_MINOR_VERSION\s+([0-9]+)').search
    patch_exp_search = re.compile(r'define\s+PG_PATCH_VERSION\s+([0-9]+)').search
    pg_header = os.path.join('src_c', 'include', '_pygame.h')
    with open(pg_header) as f:
        for line in f:
            if pgh_major == -1:
                m = major_exp_search(line)
                if m: pgh_major = int(m.group(1))
            if pgh_minor == -1:
                m = minor_exp_search(line)
                if m: pgh_minor = int(m.group(1))
            if pgh_patch == -1:
                m = patch_exp_search(line)
                if m: pgh_patch = int(m.group(1))
    if pgh_major == -1:
        raise SystemExit("_pygame.h: cannot find PG_MAJOR_VERSION")
    if pgh_minor == -1:
        raise SystemExit("_pygame.h: cannot find PG_MINOR_VERSION")
    if pgh_patch == -1:
        raise SystemExit("_pygame.h: cannot find PG_PATCH_VERSION")
    return (pgh_major, pgh_minor, pgh_patch)

def write_version_module(pygame_version, revision):
    vernum = parse_version(pygame_version)
    src_vernum = parse_source_version()
    if vernum != ', '.join(str(e) for e in src_vernum):
        raise SystemExit("_pygame.h version differs from 'METADATA' version"
                         ": %s vs %s" % (vernum, src_vernum))
    with open(os.path.join('buildconfig', 'version.py.in'), 'r') as header_file:
        header = header_file.read()
    with open(os.path.join('src_py', 'version.py'), 'w') as version_file:
        version_file.write(header)
        version_file.write('ver = "' + pygame_version + '"  # pylint: disable=invalid-name\n')
        version_file.write('vernum = PygameVersion(%s)\n' % vernum)
        version_file.write('rev = "' + revision + '"  # pylint: disable=invalid-name\n')
        version_file.write('\n__all__ = ["SDL", "ver", "vernum", "rev"]\n')

write_version_module(METADATA['version'], revision)

#required. This will be filled if doing a Windows build.
cmdclass = {}

def add_command(name):
    def decorator(command):
        assert issubclass(command, distutils.cmd.Command)
        cmdclass[name]=command
        return command
    return decorator

#try to find DLLs and copy them too  (only on windows)
if sys.platform == 'win32':

    from distutils.command.build_ext import build_ext

    #add dependency DLLs to the project
    lib_dependencies = {}
    for e in extensions:
        if e.name.startswith('COPYLIB_'):
            lib_dependencies[e.name[8:]] = e.libraries

    def dependencies(roots):
        """Return a set of dependencies for the list of library file roots

        The return set is a dictionary keyed on library root name with values of 1.
        """

        root_set = {}
        for root in roots:
            try:
                deps = lib_dependencies[root]
            except KeyError:
                pass
            else:
                root_set[root] = 1
                root_set.update(dependencies(deps))
        return root_set

    the_dlls = {}
    required_dlls = {}
    for e in extensions:
        if e.name.startswith('COPYLIB_'):
            the_dlls[e.name[8:]] = e.library_dirs[0]
        else:
            required_dlls.update(dependencies(e.libraries))

    # join the required_dlls and the_dlls keys together.
    lib_names = {}
    for lib in list(required_dlls.keys()) + list(the_dlls.keys()):
        lib_names[lib] = 1

    for lib in lib_names.keys():
        #next DLL; a distutils bug requires the paths to have Windows separators
        f = the_dlls[lib].replace('/', os.sep)
        if f == '_':
            print ("WARNING, DLL for %s library not found." % lib)
        else:
            pygame_data_files.append(f)


    if '-enable-msvc-analyze' in sys.argv:
        # calculate the MSVC compiler version as an int
        msc_pos = sys.version.find('MSC v.')
        msc_ver = 1900
        if msc_pos != -1:
            msc_ver = int(sys.version[msc_pos + 6:msc_pos + 10])
        print ('Analyzing with MSC_VER =', msc_ver)

        # excluding system headers from analyze out put was only added after MSCV_VER 1913
        if msc_ver >= 1913:
            os.environ['CAExcludePath'] = 'C:\\Program Files (x86)\\'
            for e in extensions:
                e.extra_compile_args += ['/analyze', '/experimental:external',
                                         '/external:W0', '/external:env:CAExcludePath' ]
        else:
            for e in extensions:
                e.extra_compile_args += ['/analyze']

    def has_flag(compiler, flagname):
        """
        Adapted from here: https://github.com/pybind/python_example/blob/master/setup.py#L37
        """
        from distutils.errors import CompileError
        import tempfile
        root_drive = os.path.splitdrive(sys.executable)[0] + '\\'
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
            f.write('int main (int argc, char **argv) { return 0; }')
            fname = f.name
        try:
            compiler.compile([fname], output_dir=root_drive, extra_postargs=[flagname])
        except CompileError:
            return False
        else:
            try:
                base_file = os.path.splitext(fname)[0]
                obj_file = base_file + '.obj'
                os.remove(obj_file)
            except OSError:
                pass
        finally:
            try:
                os.remove(fname)
            except OSError:
                pass
        return True

    # filter flags, returns list of accepted flags
    def flag_filter(compiler, *flags):
        return [flag for flag in flags if has_flag(compiler, flag)]

    @add_command('build_ext')
    class WinBuildExt(build_ext):
        """This build_ext sets necessary environment variables for MinGW"""

        # __sdl_lib_dir is possible location of msvcrt replacement import
        # libraries, if they exist. Pygame module base only links to SDL so
        # should have the SDL library directory as its only -L option.
        for e in extensions:
            if e.name == 'base':
                __sdl_lib_dir = e.library_dirs[0].replace('/', os.sep)
                break

        def build_extensions(self):
            # Add supported optimisations flags to reduce code size with MSVC
            opts = flag_filter(self.compiler, "/GF", "/Gy")
            for extension in extensions:
                extension.extra_compile_args += opts

            build_ext.build_extensions(self)

    # Add the precompiled smooth scale MMX functions to transform.
    def replace_scale_mmx():
        for e in extensions:
            if e.name == 'transform':
                if '64 bit' in sys.version:
                    e.extra_objects.append(
                        os.path.join('buildconfig', 'obj', 'win64', 'scale_mmx.obj'))
                else:
                    e.extra_objects.append(
                        os.path.join('buildconfig', 'obj', 'win32', 'scale_mmx.obj'))
                for i in range(len(e.sources)):
                    if e.sources[i].endswith('scale_mmx.c'):
                        del e.sources[i]
                        return
    replace_scale_mmx()


#clean up the list of extensions
for e in extensions[:]:
    if e.name.startswith('COPYLIB_'):
        extensions.remove(e) #don't compile the COPYLIBs, just clean them
    else:
        e.name = 'pygame.' + e.name #prepend package name on modules


#data installer with improved intelligence over distutils
#data files are copied into the project directory instead
#of willy-nilly
@add_command('install_data')
class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the actual library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)


@add_command('sdist')
class OurSdist(sdist):
    def initialize_options(self):
        sdist.initialize_options(self)
        # we do not want MANIFEST.in to appear in the root cluttering up things.
        self.template = os.path.join('buildconfig', 'MANIFEST.in')


if "bdist_msi" in sys.argv:
    # if you are making an msi, we want it to overwrite files
    # we also want to include the repository revision in the file name
    from distutils.command import bdist_msi
    import msilib

    @add_command('bdist_msi')
    class bdist_msi_overwrite_on_install(bdist_msi.bdist_msi):
        def run(self):
            bdist_msi.bdist_msi.run(self)

            # Remove obsolete files.
            comp = "pygame1"  # Pygame component
            prop = comp   # Directory property
            records = [("surfarray.pyd", comp,
                        "SURFAR~1.PYD|surfarray.pyd", prop, 1),
                       ("sndarray.pyd", comp,
                        "SNDARRAY.PYD|sndarray.pyd", prop, 1),
                       ("camera.pyd", comp, "CAMERA.PYD|camera.pyd", prop, 1),
                       ("color.py", comp, "COLOR.PY|color.py", prop, 1),
                       ("color.pyc", comp, "COLOR.PYC|color.pyc", prop, 1),
                       ("color.pyo", comp, "COLOR.PYO|color.pyo", prop, 1)]
            msilib.add_data(self.db, "RemoveFile", records)

            # Overwrite outdated files.
            fullname = self.distribution.get_fullname()
            installer_name = self.get_installer_filename(fullname)
            print ("changing %s to overwrite files on install" % installer_name)
            msilib.add_data(self.db, "Property", [("REINSTALLMODE", "amus")])
            self.db.Commit()

        def get_installer_filename(self, fullname):
            if revision:
                fullname += '-hg_' + revision
            return bdist_msi.bdist_msi.get_installer_filename(self, fullname)


# test command.  For doing 'python setup.py test'

@add_command('test')
class TestCommand(Command):
    user_options = [ ]

    def initialize_options(self):
        self._dir = os.getcwd()

    def finalize_options(self):
        pass

    def run(self):
        '''
        runs the tests with default options.
        '''
        import subprocess
        return subprocess.call([sys.executable, os.path.join('test', '__main__.py')])



@add_command('docs')
class DocsCommand(Command):
    """ For building the pygame documentation with `python setup.py docs`.

    This generates html, and documentation .h header files.
    """
    user_options = [ ]

    def initialize_options(self):
        self._dir = os.getcwd()

    def finalize_options(self):
        pass

    def run(self):
        '''
        runs the tests with default options.
        '''
        docs_help = (
            "Building docs requires Python version 3.6 or above, and sphinx."
        )
        if not hasattr(sys, 'version_info') or sys.version_info < (3, 6):
            raise SystemExit(docs_help)

        import subprocess
        try:
            return subprocess.call([
                sys.executable, os.path.join('buildconfig', 'makeref.py')]
            )
        except:
            print(docs_help)
            raise

# Prune empty file lists.
data_files = [(path, files) for path, files in data_files if files]

#finally,
#call distutils with all needed info
PACKAGEDATA = {
       "cmdclass":    cmdclass,
       "packages":    ['pygame',
                       'pygame.threads',
                       'pygame._sdl2',
                       'pygame.tests',
                       'pygame.tests.test_utils',
                       'pygame.tests.run_tests__tests',
                       'pygame.tests.run_tests__tests.all_ok',
                       'pygame.tests.run_tests__tests.failures1',
                       'pygame.tests.run_tests__tests.incomplete',
                       'pygame.tests.run_tests__tests.infinite_loop',
                       'pygame.tests.run_tests__tests.print_stderr',
                       'pygame.tests.run_tests__tests.print_stdout',
                       'pygame.tests.run_tests__tests.incomplete_todo',
                       'pygame.tests.run_tests__tests.exclude',
                       'pygame.tests.run_tests__tests.timeout',
                       'pygame.tests.run_tests__tests.everything',
                       'pygame.docs',
                       'pygame.examples',
                       'pygame.__pyinstaller'],
       "package_dir": {'pygame': 'src_py',
                       'pygame._sdl2': 'src_py/_sdl2',
                       'pygame.threads': 'src_py/threads',
                       'pygame.tests': 'test',
                       'pygame.docs': 'docs',
                       'pygame.examples': 'examples',
                       'pygame.__pyinstaller': 'src_py/__pyinstaller'},
       "headers":     headers,
       "ext_modules": extensions,
       "data_files":  data_files,
       "zip_safe":  False,
}
if STRIPPED:
    pygame_data_files = []
    data_files = [('pygame', ["src_py/freesansbold.ttf",
                              "src_py/pygame.ico",
                              "src_py/pygame_icon.icns",
                              "src_py/pygame_icon.svg",
                              "src_py/pygame_icon.bmp",
                              "src_py/pygame_icon.tiff"])]
    

    PACKAGEDATA = {
    "cmdclass":    cmdclass,
    "packages":    ['pygame',
                    'pygame.threads',
                    'pygame._sdl2'],
    "package_dir": {'pygame': 'src_py',
                    'pygame._sdl2': 'src_py/_sdl2',
                    'pygame.threads': 'src_py/threads'},
    "ext_modules": extensions,
    "zip_safe":  False,
    "data_files": data_files
}

PACKAGEDATA.update(METADATA)
PACKAGEDATA.update(EXTRAS)

try:
    setup(**PACKAGEDATA)
except:
    compilation_help()
    raise
