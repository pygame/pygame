# dll.py module

"""DLL specifics"""

# Some definitions:
#   Library name (name): An internal identifier, a string, for a library.
#       e.g. FONT
#   Library file root (root): The library root used in linker -l options.
#       e.g. SDL_mixer
   
import re

# Table of dependencies.
# name, root, File regex, Dependency list of file root names
libraries = [
    ('MIXER', 'SDL_mixer', r'(lib){0,1}SDL_mixer\.dll$',
     ['SDL', 'VORBISFILE', 'SMPEG']),
    ('VORBISFILE', 'vorbisfile',  r'(lib){0,1}vorbisfile(-3){0,1}\.dll$',
     ['VORBIS']),
    ('VORBIS', 'vorbis', r'(lib){0,1}vorbis(-0){0,1}\.dll$', ['OGG']),
    ('OGG', 'ogg', r'(lib){0,1}ogg(-0){0,1}\.dll$', []),
    ('SMPEG', 'smpeg', r'(lib){0,1}smpeg\.dll$', ['SDL']),
    ('IMAGE', 'SDL_image', r'(lib){0,1}SDL_image\.dll$',
     ['SDL', 'JPEG', 'PNG', 'TIFF']),
    ('TIFF', 'tiff', r'(lib){0,1}tiff\.dll$',  ['JPEG', 'Z']),
    ('JPEG', 'jpeg', r'(lib){0,1}jpeg\.dll$', []),
    ('PNG', 'png12', r'(lib){0,1}png(1[23])(-0){0,1}\.dll$', ['Z']),
#    ('FONT', 'SDL_ttf', r'(lib){0,1}SDL_ttf\.dll$', ['FREETYPE', 'SDL']),
    ('FONT', 'SDL_ttf', r'(lib){0,1}SDL_ttf\.dll$', ['SDL']),
#    ('FREETYPE', 'freetype', r'(lib){0,1}freetype(-6){0,1}\.dll$', ['Z']),
    ('Z', 'z', r'zlib1\.dll$', []),
    ('SDL', 'SDL', r'(lib){0,1}SDL\.dll$', [])
]

# regexs: Maps name to DLL file name regex.
# lib_dependencies: Maps name to list of dependencies.
# file_root_names: Maps name to root.

regexs = {}
lib_dependencies = {}
file_root_names = {}
for name, root, regex, deps in libraries:
    regexs[name] = regex
    lib_dependencies[name] = deps
    file_root_names[name] = root
del name, root, regex, deps

def tester(name):
    """For a library name return a function which tests dll file names"""
    
    def test(file_name):
        """Return true if file name f is a valid DLL name"""
        
        return match(file_name) is not None

    match =  re.compile(regexs[name], re.I).match
    test.library_name = name  # Available for debugging.
    return test

def dependencies(names):
    """Return a set of dependencies for the list of library names

    The return set is a dictionary keyed on library name with values of 1.
    """

    name_set = {}
    for name in names:
        try:
            deps = lib_dependencies[name]
        except KeyError:
            pass
        else:
            name_set[name] = 1
            name_set.update(dependencies(deps))
    return name_set

def name_to_root(name):
    """Return the library file root for the library name"""
    
    return file_root_names[name]
