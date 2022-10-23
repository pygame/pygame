#!/usr/bin/env python
#
# This is the distutils setup script for pygame.
# Full instructions are in https://www.pygame.org/wiki/GettingStarted
#
# To configure, compile, install, just run this script.
#     python setup.py install

import platform
import sysconfig

with open('README.rst', encoding='utf-8') as readme:
    LONG_DESCRIPTION = readme.read()

EXTRAS = {}

METADATA = {
    "name": "pygame",
    "version": "2.1.3",
    "license": "LGPL",
    "url": "https://www.pygame.org",
    "author": "A community project.",
    "author_email": "pygame@pygame.org",
    "description": "Python Game Development",
    "long_description": LONG_DESCRIPTION,
    "long_description_content_type": "text/x-rst",
    "project_urls": {
        "Documentation": "https://pygame.org/docs",
        "Bug Tracker": "https://github.com/pygame/pygame/issues",
        "Source": "https://github.com/pygame/pygame",
        "Twitter": "https://twitter.com/pygame_org",
    },
    "classifiers": [
        "Development Status :: 6 - Mature",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Programming Language :: Assembly",
        "Programming Language :: C",
        "Programming Language :: Cython",
        "Programming Language :: Objective C",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Games/Entertainment",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "Topic :: Multimedia :: Sound/Audio :: Players",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "Topic :: Multimedia :: Graphics :: Capture :: Screen Capture",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
    "python_requires": '>=3.6',
}

import re
import sys
import os

# just import these always and fail early if not present
from setuptools import setup
import distutils

import distutils.ccompiler

avx2_filenames = ['simd_blitters_avx2']

compiler_options = {
    'unix': ('-mavx2',),
    'msvc': ('/arch:AVX2',)
}


def spawn(self, cmd, **kwargs):
    should_use_avx2 = False
    # try to be thorough in detecting that we are on a platform that potentially supports AVX2
    machine_name = platform.machine()
    if ((machine_name.startswith(("x86", "i686")) or
        machine_name.lower() == "amd64") and
            os.environ.get("MAC_ARCH") != "arm64"):
        should_use_avx2 = True

    if should_use_avx2:
        extra_options = compiler_options.get(self.compiler_type)
        if extra_options is not None:
            # filenames are closer to the end of command line
            for argument in reversed(cmd):
                # Check if argument contains a filename. We must check for all
                # possible extensions; checking for target extension is faster.
                if not argument.endswith(self.obj_extension):
                    continue

                # check for a filename only to avoid building a new string
                # with variable extension
                for filename in avx2_filenames:
                    off_end = -len(self.obj_extension)
                    off_start = -len(filename) + off_end
                    if argument.endswith(filename, off_start, off_end):
                        if self.compiler_type == 'bcpp':
                            # Borland accepts a source file name at the end,
                            # insert the options before it
                            cmd[-1:-1] = extra_options
                        else:
                            cmd += extra_options

                # filename is found, no need to search any further
                break

    distutils.ccompiler.spawn(cmd, dry_run=self.dry_run, **kwargs)


distutils.ccompiler.CCompiler.__spawn = distutils.ccompiler.CCompiler.spawn
distutils.ccompiler.CCompiler.spawn = spawn

# A (bit hacky) fix for https://github.com/pygame/pygame/issues/2613
# This is due to the fact that distutils uses command line args to 
# export PyInit_* functions on windows, but those functions are already exported
# and that is why compiler gives warnings
from distutils.command.build_ext import build_ext

build_ext.get_export_symbols = lambda self, ext: []

IS_PYPY = '__pypy__' in sys.builtin_module_names


def compilation_help():
    """ On failure point people to a web page for help.
    """
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
        'Windows': 'https://www.pygame.org/wiki/CompileWindows',
        'Darwin': 'https://www.pygame.org/wiki/MacCompile',
        'RedHat': 'https://www.pygame.org/wiki/CompileRedHat',
        # TODO There is nothing in the following pages yet
        'Suse': 'https://www.pygame.org/wiki/CompileSuse',
        'Python (from pypy.org)': 'https://www.pygame.org/wiki/CompilePyPy',
        'Free BSD': 'https://www.pygame.org/wiki/CompileFreeBSD',
        'Debian': 'https://www.pygame.org/wiki/CompileDebian',
    }

    default = 'https://www.pygame.org/wiki/Compilation'
    url = help_urls.get(the_system, default)

    if IS_PYPY:
        url += '\n    https://www.pygame.org/wiki/CompilePyPy'

    print('\n---')
    print('For help with compilation see:')
    print(f'    {url}')
    print('To contribute to pygame development see:')
    print('    https://www.pygame.org/contribute.html')
    print('---\n')


if not hasattr(sys, 'version_info') or sys.version_info < (3, 6):
    compilation_help()
    raise SystemExit("Pygame requires Python3 version 3.6 or above.")
if IS_PYPY and sys.pypy_version_info < (7,):
    raise SystemExit("Pygame requires PyPy version 7.0.0 above, compatible with CPython >= 3.6")


def consume_arg(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


# get us to the correct directory
path = os.path.split(os.path.abspath(sys.argv[0]))[0]
os.chdir(path)

STRIPPED = False

# STRIPPED builds don't have developer resources like docs or tests
if "PYGAME_ANDROID" in os.environ:
    # test cases and docs are useless inside an APK
    STRIPPED = True

if consume_arg('-stripped'):
    STRIPPED = True

enable_arm_neon = False
if consume_arg('-enable-arm-neon'):
    enable_arm_neon = True
    cflags = os.environ.get('CFLAGS', '')
    if cflags:
        cflags += ' '
    cflags += '-mfpu=neon'
    os.environ['CFLAGS'] = cflags

compile_cython = False
cython_only = False
if consume_arg('cython'):
    compile_cython = True

if consume_arg('cython_only'):
    compile_cython = True
    cython_only = True

if compile_cython:
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
        pyx_file = ext.sources[0]  # TODO: check all sources, extension

        c_file = os.path.splitext(pyx_file)[0].split(os.path.sep)
        del c_file[1:3]  # output in src_c/
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
            print(f'Compiling {pyx_file} because it changed.')
            queue.append((priority, dict(pyx_file=pyx_file, c_file=c_file, fingerprint=None, quiet=False,
                                         options=c_options, full_module_name=ext.name,
                                         embedded_metadata=pyx_meta.get(ext.name))))

    # compile in right order
    queue.sort(key=lambda a: a[0])
    queue = [pair[1] for pair in queue]

    count = len(queue)
    for i, kwargs in enumerate(queue):
        kwargs['progress'] = f'[{i + 1}/{count}] '
        cythonize_one(**kwargs)
    
    if cython_only:
        sys.exit(0)

no_compilation = any(x in ['lint', 'format', 'docs'] for x in sys.argv)
AUTO_CONFIG = not os.path.isfile('Setup') and not no_compilation
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
        # 'install_requires': ['pyobjc'],
        # 'dependency_links': ['http://rene.f0o.com/~rene/stuff/macosx/']
    })

# headers to install
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
            destdir = os.path.join(self.install_dir, os.path.basename(header))
            self.mkpath(destdir)
            for entry in os.listdir(header):
                header1 = os.path.join(header, entry)
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

# sanity check for any arguments
if len(sys.argv) == 1 and sys.stdout.isatty():
    reply = input('\nNo Arguments Given, Perform Default Install? [Y/n]')
    if not reply or reply[0].lower() != 'n':
        sys.argv.append('install')

# make sure there is a Setup file
if AUTO_CONFIG:
    print('\n\nWARNING, No "Setup" File Exists, Running "buildconfig/config.py"')
    import buildconfig.config

    try:
        buildconfig.config.main(AUTO_CONFIG)
    except:
        compilation_help()
        raise
    if '-config' in sys.argv:
        sys.exit(0)
    print('\nContinuing With "setup.py"')

try:
    s_mtime = os.stat("Setup")[stat.ST_MTIME]
    sin_mtime = os.stat(os.path.join('buildconfig', 'Setup.SDL2.in'))[stat.ST_MTIME]
    if sin_mtime > s_mtime:
        print('\n\nWARNING, "buildconfig/Setup.SDL2.in" newer than "Setup",'
              'you might need to modify "Setup".')
except OSError:
    pass

if no_compilation:
    extensions = []
else:
    # get compile info for all extensions
    try:
        extensions = read_setup_file('Setup')
    except:
        print("""Error with the "Setup" file,
    perhaps make a clean copy from "Setup.in".""")
        compilation_help()
        raise

for e in extensions:
    # Only define the ARM_NEON defines if they have been enabled at build time.
    if enable_arm_neon:
        e.define_macros.append(('PG_ENABLE_ARM_NEON', '1'))

    e.extra_compile_args.extend(
        # some warnings are skipped here
        ("/W3", "/wd4142", "/wd4996")
        if sys.platform == "win32"
        else ("-Wall", "-Wno-error=unknown-pragmas")
    )

    if "surface" in e.name and sys.platform == "darwin":
        # skip -Werror on alphablit because sse2neon is used on arm mac
        continue

    if "freetype" in e.name and sys.platform not in ("darwin", "win32"):
        # TODO: fix freetype issues here
        if sysconfig.get_config_var("MAINCC") != "clang":        
            e.extra_compile_args.append("-Wno-error=unused-but-set-variable")

    if "mask" in e.name and sys.platform == "win32":
        # skip analyze warnings that pop up a lot in mask for now. TODO fix
        e.extra_compile_args.extend(("/wd6385", "/wd6386"))

    if (
            "CI" in os.environ
            and not e.name.startswith("_sdl2")
            and e.name not in ("pypm", "_sprite", "gfxdraw")
    ):
        # Do -Werror only on CI, and exclude -Werror on Cython C files and gfxdraw
        e.extra_compile_args.append("/WX" if sys.platform == "win32" else "-Werror")

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

# extra files to install
data_path = os.path.join(distutils.sysconfig.get_python_lib(), 'pygame')
pygame_data_files = []
data_files = [('pygame', pygame_data_files)]

# add *.pyi files into distribution directory
stub_dir = os.path.join('buildconfig', 'stubs', 'pygame')
pygame_data_files.append(os.path.join(stub_dir, 'py.typed'))
type_files = glob.glob(os.path.join(stub_dir, '*.pyi'))
for type_file in type_files:
    pygame_data_files.append(type_file)

_sdl2 = glob.glob(os.path.join(stub_dir, '_sdl2', '*.pyi'))
if _sdl2:
    _sdl2_data_files = []
    data_files.append(('pygame/_sdl2', _sdl2_data_files))
    for type_file in _sdl2:
        _sdl2_data_files.append(type_file)

# add non .py files in lib directory
for f in glob.glob(os.path.join('src_py', '*')):
    if not f[-3:] == '.py' and not f[-4:] == '.doc' and os.path.isfile(f):
        pygame_data_files.append(f)

# We don't need to deploy tests, example code, or docs inside a game

# tests/fixtures
add_datafiles(data_files, 'pygame/tests',
              ['test',
               [['fixtures',
                 [['xbm_cursors',
                   ['*.xbm']],
                  ['fonts',
                   ['*.ttf', '*.otf', '*.bdf', '*.png']]]]]])

# examples
add_datafiles(data_files, 'pygame/examples',
              ['examples', ['README.rst', ['data', ['*']]]])

# docs
add_datafiles(data_files, 'pygame/docs/generated',
              ['docs/generated',
               ['*.html',  # Navigation and help pages
                '*.txt',  # License text
                '*.js',  # For doc search
                'LGPL.txt',  # pygame license
                ['ref',  # pygame reference
                 ['*.html',  # Reference pages
                  '*.js',  # Comments script
                  '*.json']],  # Comment data
                ['c_api',  # pygame C API
                 ['*.html']],
                ['tut',  # Tutorials
                 ['*.html',
                  ['tom',
                   ['*.html',
                    '*.png']]]],
                ['_static',  # Sphinx added support files
                 ['*.css',
                  '*.png',
                  '*.ico',
                  '*.js',
                  '*.zip',
                  '*.svg']],
                ['_images',  # Sphinx added reST ".. image::" refs
                 ['*.jpg',
                  '*.png',
                  '*.gif']],
                ['_sources',  # Used for ref search
                 ['*.txt',
                  ['ref',
                   ['*.txt']]]]]])


# generate the version module
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
    with open(os.path.join('buildconfig', 'version.py.in')) as header_file:
        header = header_file.read()
    with open(os.path.join('src_py', 'version.py'), 'w') as version_file:
        version_file.write(header)
        version_file.write('ver = "' + pygame_version + '"  # pylint: disable=invalid-name\n')
        version_file.write(f'vernum = PygameVersion({vernum})\n')
        version_file.write('rev = "' + revision + '"  # pylint: disable=invalid-name\n')
        version_file.write('\n__all__ = ["SDL", "ver", "vernum", "rev"]\n')


write_version_module(METADATA['version'], revision)

# required. This will be filled if doing a Windows build.
cmdclass = {}


def add_command(name):
    def decorator(command):
        assert issubclass(command, distutils.cmd.Command)
        cmdclass[name] = command
        return command

    return decorator


# try to find DLLs and copy them too  (only on windows)
if sys.platform == 'win32' and not 'WIN32_DO_NOT_INCLUDE_DEPS' in os.environ:

    from distutils.command.build_ext import build_ext

    # add dependency DLLs to the project
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
        # next DLL; a distutils bug requires the paths to have Windows separators
        f = the_dlls[lib].replace('/', os.sep)
        if f == '_':
            print(f"WARNING, DLL for {lib} library not found.")
        else:
            pygame_data_files.append(f)

    if '-enable-msvc-analyze' in sys.argv:
        # calculate the MSVC compiler version as an int
        msc_pos = sys.version.find('MSC v.')
        msc_ver = 1900
        if msc_pos != -1:
            msc_ver = int(sys.version[msc_pos + 6:msc_pos + 10])
        print('Analyzing with MSC_VER =', msc_ver)

        # excluding system headers from analyze out put was only added after MSCV_VER 1913
        if msc_ver >= 1913:
            os.environ['CAExcludePath'] = 'C:\\Program Files (x86)\\'

        for e in extensions:
            e.extra_compile_args.extend(
                (
                    "/analyze",
                    "/wd28251",
                    "/wd28301",
                )
            )

            if msc_ver >= 1913:
                e.extra_compile_args.extend(
                    (
                        "/experimental:external",
                        "/external:W0",
                        "/external:env:CAExcludePath",
                    )
                )


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


    # Only on win32, not MSYS2
    if 'MSYSTEM' not in os.environ:
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

# clean up the list of extensions
for e in extensions[:]:
    if e.name.startswith('COPYLIB_'):
        extensions.remove(e)  # don't compile the COPYLIBs, just clean them
    else:
        e.name = 'pygame.' + e.name  # prepend package name on modules


# data installer with improved intelligence over distutils
# data files are copied into the project directory instead
# of willy-nilly
@add_command('install_data')
class smart_install_data(install_data):
    def run(self):
        # need to change self.install_dir to the actual library dir
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
            prop = comp  # Directory property
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
            print(f"changing {installer_name} to overwrite files on install")
            msilib.add_data(self.db, "Property", [("REINSTALLMODE", "amus")])
            self.db.Commit()

        def get_installer_filename(self, fullname):
            if revision:
                fullname += '-hg_' + revision
            return bdist_msi.bdist_msi.get_installer_filename(self, fullname)


# test command.  For doing 'python setup.py test'

@add_command('test')
class TestCommand(Command):
    user_options = []

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


class LintFormatCommand(Command):
    """ Used for formatting or linting. See Lint and Format Sub classes.
    """
    user_options = []
    lint = False
    format = False

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Check the existence and launch linters."""
        import subprocess
        import sys
        import warnings
        import pathlib

        def check_linter_exists(linter):
            if shutil.which(linter) is None:
                msg = "Please install '%s' in your environment. (hint: 'python3 -m pip install %s')"
                warnings.warn(msg % (linter, linter))
                sys.exit(1)

        def filter_files(path_obj, all_files, allowed_files, disallowed_files):
            files = []
            for file in all_files:
                for disallowed in disallowed_files:
                    if file.match(str(path_obj / disallowed)):
                        break
                else:  # no-break
                    files.append(str(file))
                    continue

                for allowed in allowed_files:
                    if file.match(str(path_obj / allowed)):
                        files.append(str(file))
                        break

            return files

        path_obj = pathlib.Path(path, "src_c")
        c_files_unfiltered = path_obj.glob("**/*.[ch]")
        c_file_disallow = [
            "_sdl2/**",
            "pypm.c",
            "SDL_gfx/**",
            "**/sse2neon.h",
            "doc/**",
            "_sprite.c",
        ]
        c_file_allow = ["_sdl2/touch.c"]
        c_files = filter_files(path_obj, c_files_unfiltered, c_file_allow, c_file_disallow)


        # Other files have too many issues for now. setup.py, buildconfig, etc
        python_directories = ["src_py", "test", "examples"]
        if self.lint:
            commands = {
                "clang-format": ["--dry-run", "--Werror", "-i"] + c_files,
                "black": ["--check", "--diff"] + python_directories,
                # Test directory has too much pylint warning for now
                "pylint": ["src_py"],
            }
        else:
            commands = {
                "clang-format": ["-i"] + c_files,
                "black": python_directories,
            }

        formatters = ["black", "clang-format"]
        for linter, option in commands.items():
            print(" ".join([linter] + option))
            check_linter_exists(linter)
            result = subprocess.run([linter] + option)
            if result.returncode:
                msg = f"'{linter}' failed."
                msg += " Please run: python setup.py format" if linter in formatters else ""
                msg += f" Do you have the latest version of {linter}?"
                raise SystemExit(msg)


@add_command("lint")
class LintCommand(LintFormatCommand):
    lint = True


@add_command("format")
class FormatCommand(LintFormatCommand):
    format = True


@add_command('docs')
class DocsCommand(Command):
    """ For building the pygame documentation with `python setup.py docs`.
    This generates html, and documentation .h header files.
    """
    user_options = [
        (
            'fullgeneration',
            'f',
            'Full generation. Do not use a saved environment, always read all files.'
        )
    ]
    boolean_options = ['fullgeneration']

    def initialize_options(self):
        self._dir = os.getcwd()
        self.fullgeneration = None

    def finalize_options(self):
        pass

    def run(self):
        '''
        runs Sphinx to build the docs.
        '''
        import subprocess
        print("Using python:", sys.executable)
        command_line = [
            sys.executable, os.path.join('buildconfig', 'makeref.py')
        ]
        if self.fullgeneration:
            command_line.append('full_generation')
        if subprocess.call(command_line) != 0:
            raise SystemExit("Failed to build documentation")


# Prune empty file lists.
data_files = [(path, files) for path, files in data_files if files]

# finally,
# call distutils with all needed info
PACKAGEDATA = {
    "cmdclass": cmdclass,
    "packages": ['pygame',
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
    "headers": headers,
    "ext_modules": extensions,
    "data_files": data_files,
    "zip_safe": False,
}
if STRIPPED:
    pygame_data_files = []
    data_files = [('pygame', ["src_py/freesansbold.ttf",
                              "src_py/pygame.ico",
                              "src_py/pygame_icon.icns",
                              "src_py/pygame_icon.bmp",
                              "src_py/pygame_icon_mac.bmp"])]

    PACKAGEDATA = {
        "cmdclass": cmdclass,
        "packages": ['pygame',
                     'pygame.threads',
                     'pygame._sdl2'],
        "package_dir": {'pygame': 'src_py',
                        'pygame._sdl2': 'src_py/_sdl2',
                        'pygame.threads': 'src_py/threads'},
        "ext_modules": extensions,
        "zip_safe": False,
        "data_files": data_files
    }

PACKAGEDATA.update(METADATA)
PACKAGEDATA.update(EXTRAS)

try:
    setup(**PACKAGEDATA)
except:
    compilation_help()
    raise
