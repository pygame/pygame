# coding: ascii
# pygame - Python Game Library
# Copyright (C) 2000-2003  Pete Shinners
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the Free
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Pete Shinners
# pete@shinners.org
"""sysfont, used in the font module to find system fonts"""

import os
import sys
from pygame.compat import xrange_, PY_MAJOR_VERSION
from os.path import basename, dirname, exists, join, splitext


OpenType_extensions = frozenset(('.ttf', '.ttc', '.otf'))
Sysfonts = {}
Sysalias = {}

# Python 3 compatibility
if PY_MAJOR_VERSION >= 3:
    def toascii(raw):
        """convert bytes to ASCII-only string"""
        return raw.decode('ascii', 'ignore')
    if os.name == 'nt':
        import winreg as _winreg
    else:
        import subprocess
else:
    def toascii(raw):
        """return ASCII characters of a given unicode or 8-bit string"""
        return raw.decode('ascii', 'ignore')
    if os.name == 'nt':
        import _winreg
    else:
        import subprocess


def _simplename(name):
    """create simple version of the font name"""
    # return alphanumeric characters of a string (converted to lowercase)
    return ''.join(c.lower() for c in name if c.isalnum())


def _addfont(name, bold, italic, font, fontdict):
    """insert a font and style into the font dictionary"""
    if name not in fontdict:
        fontdict[name] = {}
    fontdict[name][bold, italic] = font


def initsysfonts_win32():
    """initialize fonts dictionary on Windows"""

    fontdir = join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')

    TrueType_suffix = '(TrueType)'
    mods = ('demibold', 'narrow', 'light', 'unicode', 'bt', 'mt')

    fonts = {}

    # add fonts entered in the registry

    # find valid registry keys containing font information.
    # http://docs.python.org/lib/module-sys.html
    # 0 (VER_PLATFORM_WIN32s)          Win32s on Windows 3.1
    # 1 (VER_PLATFORM_WIN32_WINDOWS)   Windows 95/98/ME
    # 2 (VER_PLATFORM_WIN32_NT)        Windows NT/2000/XP
    # 3 (VER_PLATFORM_WIN32_CE)        Windows CE
    if sys.getwindowsversion()[0] == 1:
        key_name = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Fonts"
    else:
        key_name = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Fonts"
    key = _winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE, key_name)

    for i in xrange_(_winreg.QueryInfoKey(key)[1]):
        try:
            # name is the font's name e.g. Times New Roman (TrueType)
            # font is the font's filename e.g. times.ttf
            name, font = _winreg.EnumValue(key, i)[0:2]
        except EnvironmentError:
            break

        # try to handle windows unicode strings for file names with
        # international characters
        if PY_MAJOR_VERSION < 3:
            # here are two documents with some information about it:
            # http://www.python.org/peps/pep-0277.html
            # https://www.microsoft.com/technet/archive/interopmigration/linux/mvc/lintowin.mspx#ECAA
            try:
                font = str(font)
            except UnicodeEncodeError:
                # MBCS is the windows encoding for unicode file names.
                try:
                    font = font.encode('MBCS')
                except:
                    # no success with str or MBCS encoding... skip this font.
                    continue

        if splitext(font)[1].lower() not in OpenType_extensions:
            continue
        if not dirname(font):
            font = join(fontdir, font)

        if name.endswith(TrueType_suffix):
            name = name.rstrip(TrueType_suffix).rstrip()
        name = name.lower().split()

        bold = italic = 0
        for m in mods:
            if m in name:
                name.remove(m)
        if 'bold' in name:
            name.remove('bold')
            bold = 1
        if 'italic' in name:
            name.remove('italic')
            italic = 1
        name = ''.join(name)

        name = _simplename(name)

        _addfont(name, bold, italic, font, fonts)

    return fonts


def initsysfonts_darwin():
    """read the fonts on OS X. X11 is required for this to work."""
    # if the X11 binary exists... try and use that.
    #  Not likely to be there on pre 10.4.x ...
    if exists("/usr/X11/bin/fc-list"):
        fonts = initsysfonts_unix("/usr/X11/bin/fc-list")
    # This fc-list path will work with the X11 from the OS X 10.3 installation
    # disc
    elif exists("/usr/X11R6/bin/fc-list"):
        fonts = initsysfonts_unix("/usr/X11R6/bin/fc-list")
    elif exists("/usr/sbin/system_profiler"):
        fonts = initsysfonts_macos("/usr/sbin/system_profiler")
    else:
        fonts = {}

    return fonts


def _add_sys_font_inner(current_font, fonts):

    bold = 'bold' in current_font['style']
    italic = 'italic' in current_font['style']
    oblique = 'oblique' in current_font['style']

    _addfont(
        _simplename(current_font['full name']), bold, italic or oblique, current_font['path'],
        fonts)


def _add_sys_font(current_font, multiple_fonts, fonts):

    if len(multiple_fonts) > 0:
        for font_item in multiple_fonts:
            _add_sys_font_inner(font_item, fonts)

    if len(current_font) > 0:
        _add_sys_font_inner(current_font, fonts)


# read the fonts using system_profiler on macOS
def initsysfonts_macos(path="/usr/sbin/system_profiler"):
    """use system_profiler get a list of fonts"""
    fonts = {}

    arguments = "SPFontsDataType | grep -i -e 'location' -e 'family' -e 'style' -e 'name'"

    try:
        flout, flerr = subprocess.Popen('%s : %s' % (path, arguments), shell=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        close_fds=True).communicate()
    except Exception:
        return fonts

    entries = toascii(flout)

    parsing_font = False
    current_font = {}

    multiple_fonts = []

    lines = entries.split('\n')
    lines_len = len(lines);
                
    try:
        for idx in range(0, lines_len):

            try:
                key, value = lines[idx].strip().split(':', 1)

                key = key.strip()
                value = value.strip().replace(':', "")

                if not value:
                    continue

                if parsing_font and (os.path.exists(value)):

                    _add_sys_font(current_font, multiple_fonts, fonts)

                    current_font.clear()
                    multiple_fonts = []
                    parsing_font = False

                if not parsing_font and os.path.exists(value):
                    
                    if splitext(value)[1].lower() in OpenType_extensions:

                        parsing_font = True
                        current_font['path'] = value

                        continue

                if key.lower() in current_font:

                    font_path = current_font['path']

                    multiple_fonts.append(current_font.copy())
                    current_font.clear()
                    current_font['path'] = font_path
                    current_font[key.lower()] = value.lower()

                else:
                    current_font[key.lower()] = value.lower()

            except Exception:
                # try the next one.
                pass

    except Exception:
        pass

    _add_sys_font(current_font, multiple_fonts, fonts)

    return fonts


# read the fonts on unix
def initsysfonts_unix(path="fc-list"):
    """use the fc-list from fontconfig to get a list of fonts"""
    fonts = {}

    try:
        # note, we capture stderr so if fc-list isn't there to stop stderr
        # printing.
        flout, flerr = subprocess.Popen('%s : file family style' % path, shell=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        close_fds=True).communicate()
    except Exception:
        return fonts

    entries = toascii(flout)
    try:
        for line in entries.split('\n'):

            try:
                filename, family, style = line.split(':', 2)
                if splitext(filename)[1].lower() in OpenType_extensions:
                    bold = 'Bold' in style
                    italic = 'Italic' in style
                    oblique = 'Oblique' in style
                    for name in family.strip().split(','):
                        if name:
                            break
                    else:
                        name = splitext(basename(filename))[0]

                    _addfont(
                        _simplename(name), bold, italic or oblique, filename, fonts)

            except Exception:
                # try the next one.
                pass

    except Exception:
        pass

    return fonts


def create_aliases():
    """map common fonts that are absent from the system to similar fonts that are installed in the system"""
    alias_groups = (
        ('monospace', 'misc-fixed', 'courier', 'couriernew', 'console',
         'fixed', 'mono', 'freemono', 'bitstreamverasansmono',
         'verasansmono', 'monotype', 'lucidaconsole'),
        ('sans', 'arial', 'helvetica', 'swiss', 'freesans',
         'bitstreamverasans', 'verasans', 'verdana', 'tahoma'),
        ('serif', 'times', 'freeserif', 'bitstreamveraserif', 'roman',
         'timesroman', 'timesnewroman', 'dutch', 'veraserif',
         'georgia'),
        ('wingdings', 'wingbats'),
    )
    for alias_set in alias_groups:
        for name in alias_set:
            if name in Sysfonts:
                found = Sysfonts[name]
                break
        else:
            continue
        for name in alias_set:
            if name not in Sysfonts:
                Sysalias[name] = found


# initialize it all, called once
def initsysfonts():
    if sys.platform == 'win32':
        fonts = initsysfonts_win32()
    elif sys.platform == 'darwin':
        fonts = initsysfonts_darwin()
    else:
        fonts = initsysfonts_unix()
    Sysfonts.update(fonts)
    create_aliases()
    if not Sysfonts:  # dummy so we don't try to reinit
        Sysfonts[None] = None


# pygame.font specific declarations
def font_constructor(fontpath, size, bold, italic):
    import pygame.font

    font = pygame.font.Font(fontpath, size)
    if bold:
        font.set_bold(1)
    if italic:
        font.set_italic(1)

    return font


# the exported functions

def SysFont(name, size, bold=False, italic=False, constructor=None):
    """pygame.font.SysFont(name, size, bold=False, italic=False, constructor=None) -> Font
       create a pygame Font from system font resources

       This will search the system fonts for the given font
       name. You can also enable bold or italic styles, and
       the appropriate system font will be selected if available.

       This will always return a valid Font object, and will
       fallback on the builtin pygame font if the given font
       is not found.

       Name can also be a comma separated list of names, in
       which case set of names will be searched in order. Pygame
       uses a small set of common font aliases, if the specific
       font you ask for is not available, a reasonable alternative
       may be used.

       if optional contructor is provided, it must be a function with
       signature constructor(fontpath, size, bold, italic) which returns
       a Font instance. If None, a pygame.font.Font object is created.
    """
    if constructor is None:
        constructor = font_constructor

    if not Sysfonts:
        initsysfonts()

    gotbold = gotitalic = False
    fontname = None
    if name:
        allnames = name
        for name in allnames.split(','):
            name = _simplename(name)
            styles = Sysfonts.get(name)
            if not styles:
                styles = Sysalias.get(name)
            if styles:
                plainname = styles.get((False, False))
                fontname = styles.get((bold, italic))
                if not fontname and not plainname:
                    # Neither requested style, nor plain font exists, so
                    # return a font with the name requested, but an
                    # arbitrary style.
                    (style, fontname) = list(styles.items())[0]
                    # Attempt to style it as requested. This can't
                    # unbold or unitalicize anything, but it can
                    # fake bold and/or fake italicize.
                    if bold and style[0]:
                        gotbold = True
                    if italic and style[1]:
                        gotitalic = True
                elif not fontname:
                    fontname = plainname
                elif plainname != fontname:
                    gotbold = bold
                    gotitalic = italic
            if fontname:
                break

    set_bold = set_italic = False
    if bold and not gotbold:
        set_bold = True
    if italic and not gotitalic:
        set_italic = True

    return constructor(fontname, size, set_bold, set_italic)


def get_fonts():
    """pygame.font.get_fonts() -> list
       get a list of system font names

       Returns the list of all found system fonts. Note that
       the names of the fonts will be all lowercase with spaces
       removed. This is how pygame internally stores the font
       names for matching.
    """
    if not Sysfonts:
        initsysfonts()
    return list(Sysfonts)


def match_font(name, bold=0, italic=0):
    """pygame.font.match_font(name, bold=0, italic=0) -> name
       find the filename for the named system font

       This performs the same font search as the SysFont()
       function, only it returns the path to the TTF file
       that would be loaded. The font name can be a comma
       separated list of font names to try.

       If no match is found, None is returned.
    """
    if not Sysfonts:
        initsysfonts()

    fontname = None
    allnames = name
    for name in allnames.split(','):
        name = _simplename(name)
        styles = Sysfonts.get(name)
        if not styles:
            styles = Sysalias.get(name)
        if styles:
            while not fontname:
                fontname = styles.get((bold, italic))
                if italic:
                    italic = 0
                elif bold:
                    bold = 0
                elif not fontname:
                    fontname = list(styles.values())[0]
        if fontname:
            break
    return fontname
