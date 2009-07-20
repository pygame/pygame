# dll.py module

"""DLL specifics

Configured for the Pygame 1.9.0 dependencies as built by msys_build_deps.py.
"""

# Some definitions:
#   Library name (name): An internal identifier, a string, for a library.
#       e.g. FONT
#   Library file root (root): The library root used in linker -l options.
#       e.g. SDL_mixer
   
import re

# Table of dependencies.
# name, root, File regex, Dependency list of names
libraries = [
    ('MIXER', 'SDL_mixer', r'SDL_mixer\.dll$',
     ['SDL', 'VORBISFILE', 'SMPEG']),
    ('VORBISFILE', 'vorbisfile',  r'libvorbisfile-3\.dll$',
     ['VORBIS']),
    ('VORBIS', 'vorbis', r'libvorbis-0\.dll$', ['OGG']),
    ('OGG', 'ogg', r'libogg-0\.dll$', []),
    ('SMPEG', 'smpeg', r'smpeg\.dll$', ['SDL']),
    ('IMAGE', 'SDL_image', r'SDL_image\.dll$',
     ['SDL', 'JPEG', 'PNG', 'TIFF']),
    ('TIFF', 'tiff', r'libtiff\.dll$',  ['JPEG', 'Z']),
    ('JPEG', 'jpeg', r'jpeg\.dll$', []),
    ('PNG', 'png12', r'libpng12-0\.dll$', ['Z']),
    ('FONT', 'SDL_ttf', r'SDL_ttf\.dll$', ['SDL']),
    ('FREETYPE', 'freetype', r'libfreetype-6\.dll$', ['Z']),
    ('Z', 'z', r'zlib1\.dll$', []),
    ('SDL', 'SDL', r'SDL\.dll$', []),
    ('PORTMIDI', 'portmidi', r'portmidi\.dll', []),
    ('PORTTIME', 'portmidi', r'portmidi\.dll', []),
    ('AVCODEC', 'avcodec', r'avcodec-50\.dll', []),
    ('AVFORMAT', 'avformat', r'avformat-52\.dll', []),
    ('AVDEVICE', 'avdevice', r'avdevice-52\.dll', []),
    ('AVUTIL', 'avutil', r'avutil-50\.dll', []),
    ('SWSCALE', 'swscale', r'swscale-0.\dll', []),
]

# regexs: Maps name to DLL file name regex.
# lib_dependencies: Maps name to list of dependencies.
# file_root_names: Maps name to root.

regexs = {}
lib_dependencies = {}
file_root_names = {}
for name, root, ignore1, ignore2 in libraries:
    file_root_names[name] = root
for name, root, regex, deps in libraries:
    regexs[name] = regex
    lib_dependencies[root] = [file_root_names[d] for d in deps]
del name, root, regex, deps, ignore1, ignore2

def tester(name):
    """For a library name return a function which tests dll file names"""
    
    def test(file_name):
        """Return true if file name f is a valid DLL name"""
        
        return match(file_name) is not None

    match =  re.compile(regexs[name], re.I).match
    test.library_name = name  # Available for debugging.
    return test

def name_to_root(name):
    """Return the library file root for the library name"""
    
    return file_root_names[name]

def libraries(name):
    """Return the library file roots this library links too"""

    return lib_dependencies[name_to_root(name)]
