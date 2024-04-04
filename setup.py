"""

Imports

"""
import platform
import sys
import os
import glob
import re
import sysconfig
import shutil
import stat
from setuptools import setup, Command

METADATA = {'version': "2.6.0.dev1"}
revision = ''


"""

Utils

"""
def consume_arg(name):
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False

def compilation_help():
    """ On failure point people to a web page for help.
    """
    the_system = platform.system()
    if the_system == 'Linux' and hasattr(platform, 'linux_distribution'):
        distro_name = platform.linux_distribution()[0].lower()
        distro_mapping = {
            'ubuntu': 'Ubuntu',
            'debian': 'Debian'
        }
        the_system = distro_mapping.get(distro_name, the_system)

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

cmdclass = {}
def add_command(name):
    def decorator(command):
        assert issubclass(command, Command)
        cmdclass[name] = command
        return command

    return decorator


"""

Platform Constants / Flags

"""
IS_PYPY = '__pypy__' in sys.builtin_module_names

IS_MSC = sys.platform == "win32" and "MSC" in sys.version

STRIPPED = False
if "PYGAME_ANDROID" in os.environ:
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

no_compilation = any(x in ['lint', 'format', 'docs'] for x in sys.argv)
AUTO_CONFIG = not os.path.isfile('Setup') and not no_compilation
if consume_arg('-auto'):
    AUTO_CONFIG = True


"""

Cython Compilation

"""
compile_cython = False
cython_only = False
if consume_arg('cython'):
    compile_cython = True

if consume_arg('cython_only'):
    compile_cython = True
    cython_only = True

# If there is no generated C code, compile the cython/.pyx files
if any(x in ["build_ext", "build", "sdist", "bdist_wheel"] for x in sys.argv) and (
    not glob.glob(os.path.join("src_c", "_sdl2", "audio.c"))
    or not glob.glob(os.path.join("src_c", "pypm.c"))
):
    compile_cython = True
    print ("Compiling Cython files")
else:
    print ("Skipping Cython compilation")

if compile_cython:
    # compile .pyx files
    # So you can `setup.py cython` or `setup.py cython install`
    try:
        from Cython.Build.Dependencies import cythonize_one
    except ImportError:
        print("You need cython. https://cython.org/, python -m pip install cython --user")
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


"""

C Extensions (run buildconfig, read in setup file, generate list of headers)

"""
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
        from distutils.extension import read_setup_file
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
        ("/W3", "/wd4142", "/wd4996") if IS_MSC else ("-Wall", "-Wno-error=unknown-pragmas")
    )

    if "surface" in e.name and sys.platform == "darwin":
        # skip -Werror on alphablit because sse2neon is used on arm mac
        continue

    if "rwobject" in e.name and not IS_MSC:
        # because Py_FileSystemDefaultEncoding is deprecated in 3.12.0a7
        e.extra_compile_args.append("-Wno-error=deprecated-declarations")

    if "freetype" in e.name and sys.platform not in ("darwin", "win32"):
        # TODO: fix freetype issues here
        if sysconfig.get_config_var("MAINCC") != "clang":        
            e.extra_compile_args.append("-Wno-error=unused-but-set-variable")

    if "mask" in e.name and IS_MSC:
        # skip analyze warnings that pop up a lot in mask for now. TODO fix
        e.extra_compile_args.extend(("/wd6385", "/wd6386"))

    if (
            "CI" in os.environ
            and not e.name.startswith("_sdl2")
            and e.name not in ("pypm", "_sprite", "gfxdraw")
    ):
        # Do -Werror only on CI, and exclude -Werror on Cython C files and gfxdraw
        e.extra_compile_args.append("/WX" if IS_MSC else "-Wundef")

# headers to install
headers = glob.glob(os.path.join('src_c', '*.h'))
headers.remove(os.path.join('src_c', 'scale.h'))
headers.append(os.path.join('src_c', 'include'))
# option for not installing the headers.
if consume_arg("-noheaders"):
    headers = []

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


"""

Create list of data_files to be included in distribution

"""
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

# extra files to install
data_path = os.path.join(sysconfig.get_paths()['purelib'], 'pygame')
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

# if windows build, get DLLs
if sys.platform == 'win32' and not 'WIN32_DO_NOT_INCLUDE_DEPS' in os.environ:
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

# clean up the list of extensions
for e in extensions[:]:
    if e.name.startswith('COPYLIB_'):
        extensions.remove(e)  # don't compile the COPYLIBs, just clean them
    else:
        e.name = 'pygame.' + e.name  # prepend package name on modules

# Prune empty file lists.
data_files = [(path, files) for path, files in data_files if files]


"""

Custom Commands

"""
@add_command('install_headers')
class CustomInstallHeaders(Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
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
            else:
                (out, _) = self.copy_file(header, self.install_dir)

# using distutils install_data because setuptools does not work for .dll and .pyi
from distutils.command.install_data import install_data
@add_command('install_data')
class smart_install_data(install_data):
    def run(self):
        # need to change self.install_dir to the actual library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

# custom build_ext for windows
if sys.platform == 'win32' and not 'WIN32_DO_NOT_INCLUDE_DEPS' in os.environ and IS_MSC:
    from setuptools.command.build_ext import build_ext

    def has_flag(compiler, flagname):
        """
        Adapted from here: https://github.com/pybind/python_example/blob/master/setup.py#L37
        """
        from setuptools.errors import CompileError
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

"""

Misc. Platform Specific Compilation Stuff

"""
if sys.platform == 'win32' and not 'WIN32_DO_NOT_INCLUDE_DEPS' in os.environ:
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
    if IS_MSC:
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

        if not 'ARM64' in sys.version:
            replace_scale_mmx()

# avx2
if os.environ.get('PYGAME_DETECT_AVX2', '') != '':
    avx2_filenames = ['simd_blitters_avx2']
    compiler_options = {
        'unix': ('-mavx2',),
        'msvc': ('/arch:AVX2',)
    }
    # infer compiler type from os
    operating_system = platform.system()
    if operating_system == 'Windows':
        compiler_type = 'msvc'
    elif operating_system in ('Linux', 'Darwin'):
        compiler_type = 'unix'
    else:
        compiler_type = 'other'
    should_use_avx2 = False
    # try to be thorough in detecting that we are on a platform that potentially supports AVX2
    machine_name = platform.machine()
    if ((machine_name.startswith(("x86", "i686")) or
        machine_name.lower() == "amd64") and
            os.environ.get("MAC_ARCH") != "arm64"):
        should_use_avx2 = True
    
    if should_use_avx2:
        extra_options = compiler_options.get(compiler_type)
        if extra_options is not None:
            for e in extensions:
                if any([f in source for source in e.sources for f in avx2_filenames]):
                    e.extra_compile_args.append(extra_options)


"""

Version Module

"""
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


"""

Finally, build pygame!

"""
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
        "data_files": data_files,
    }

EXTRAS = {}
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

PACKAGEDATA.update(EXTRAS)
PACKAGEDATA.update(METADATA)

# we need to eat this argument in to distutils doesn't trip over it
consume_arg("-setuptools")
# sanity check for any arguments
if len(sys.argv) == 1 and sys.stdout.isatty():
    reply = input('\nNo Arguments Given, Perform Default Install? [Y/n]')
    if not reply or reply[0].lower() != 'n':
        sys.argv.append('install')

try:
    setup(**PACKAGEDATA)
    #print(PACKAGEDATA)
except:
    compilation_help()
    raise


# MISC ADDITIONS TO BE MADE:
# bonus custom commands (docs, sdist, linting/formatting, etc)
# remove distutils from buildconfig
# modify appveyor.yml